"""Gateway — the long-running daemon process for CLIver.

Hosts a cron scheduler for background tasks, a Starlette web application
for the API, and platform adapters for messaging integrations.
"""

import asyncio
import fcntl
import importlib
import logging
import os
import tempfile
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import List, Optional

from cliver.agent import Agent
from cliver.agent_profile import CliverProfile
from cliver.config import ConfigManager
from cliver.gateway.adapter_manager import AdapterManager
from cliver.gateway.adapters import BUILTIN_ADAPTERS
from cliver.gateway.platform_adapter import (
    MediaAttachment,
    MessageEvent,
    PlatformAdapter,
    split_message,
)
from cliver.gateway.scheduler import Scheduler
from cliver.gateway.task_store import TaskStore
from cliver.mcp import MCPClient
from cliver.messages import CLIverMessage
from cliver.session_manager import SessionManager
from cliver.task_manager import TaskDefinition, TaskManager, TaskRun
from cliver.tool import ToolRegistry, discover_builtin_tools

logger = logging.getLogger(__name__)

im_context: ContextVar[Optional[dict]] = ContextVar("im_context", default=None)


class ThreadQueue:
    """Per-thread asyncio locks for sequential message processing."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_used: dict[str, float] = {}

    def get_lock(self, session_key: str) -> asyncio.Lock:
        if session_key not in self._locks:
            self._locks[session_key] = asyncio.Lock()
        self._last_used[session_key] = time.monotonic()
        return self._locks[session_key]

    def cleanup(self, max_idle_seconds: float = 3600):
        now = time.monotonic()
        stale = [k for k, t in self._last_used.items() if now - t > max_idle_seconds]
        for key in stale:
            self._locks.pop(key, None)
            self._last_used.pop(key, None)


class Gateway:
    """Top-level orchestrator for the gateway daemon.

    Manages the lifecycle of:
    - Starlette web application (API server)
    - CronScheduler (tick-based task execution)
    - AdapterManager (platform adapter connections)
    """

    def __init__(
        self,
        config_dir: Path,
        resolved_config=None,
    ):
        self.config_dir = Path(config_dir)
        self._resolved_config = resolved_config
        self._pid_path = self.config_dir / "cliver-gateway.pid"
        self._pid_file = None
        self._mcp_client: Optional[MCPClient] = None
        self._builtin_tools: list = []
        self._agents: dict[str, Agent] = {}
        self._scheduler: Optional[Scheduler] = None
        self._run_store: Optional[TaskStore] = None
        self._task_manager: Optional[TaskManager] = None
        self._adapter_manager: Optional[AdapterManager] = None
        self._session_manager: Optional[SessionManager] = None
        self._key_store = None
        self._cli_session_manager: Optional[SessionManager] = None
        self._tasks_run = 0
        self._start_time = 0.0
        self._thread_queue = ThreadQueue()
        self._lab_store = None

    def init(self) -> None:
        """Initialize shared resources that need to run before fork.

        Creates MCPClient, discovers builtin tools, resolves secrets.
        Call this in the parent process before fork — the child inherits
        the initialized state and never touches the keychain.
        """
        self._init_shared_resources()

        # Initialize stores backed by the unified database
        db_path = self._agent_profile.db_path

        try:
            from cliver.key_store import KeyStore

            self._key_store = KeyStore(db_path)
            logger.info("Key store initialized")
        except Exception as e:
            logger.error("Failed to init key store: %s", e)
            self._key_store = None

        try:
            self._cli_session_manager = SessionManager(db_path)
            logger.info("CLI session manager initialized")
        except Exception as e:
            logger.error("Failed to init CLI session manager: %s", e)
            self._cli_session_manager = None

        try:
            from cliver.lab.store import LabStore

            self._lab_store = LabStore(db_path)
            logger.info("Lab store initialized")
        except Exception as e:
            logger.error("Failed to init lab store: %s", e)
            self._lab_store = None

    def _get_config_manager(self) -> "ConfigManager":
        """Get config manager, using pre-resolved config if available."""
        return ConfigManager(self.config_dir, config=self._resolved_config)

    def create_app(self):
        """Create and return the Starlette web application."""
        from contextlib import asynccontextmanager

        from starlette.applications import Starlette

        routes = self._build_routes()

        gateway_ref = self

        @asynccontextmanager
        async def lifespan(app):
            await gateway_ref._on_startup()
            yield
            await gateway_ref._on_cleanup()

        app = Starlette(
            routes=routes,
            lifespan=lifespan,
        )
        return app

    def _build_routes(self) -> list:
        """Build the full list of Starlette routes."""
        routes = []

        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def health_endpoint(request):
            return JSONResponse({"status": "ok", **self._get_status()})

        routes.append(Route("/health", health_endpoint))

        # Admin portal routes
        try:
            from cliver.gateway.admin import get_admin_routes

            gw_config = self._get_config_manager().config.gateway
            admin_user = gw_config.admin_username if gw_config else None
            admin_pass = gw_config.admin_password if gw_config else None

            config_manager = self._get_config_manager()

            admin_ctx = {
                "get_status": self._get_status_async,
                "profile_name": self._agent_profile.profile_name,
                "config_dir": self.config_dir,
                "gateway": self,
                "cli_session_manager": self._cli_session_manager,
                "config_manager": config_manager,
                "key_store": self._key_store,
            }
            # Admin API routes (returns auth function for reuse)
            admin_api_routes, spa_routes, shared_auth = get_admin_routes(
                username=admin_user, password=admin_pass, context=admin_ctx
            )

            # Lab routes
            try:
                if hasattr(self, "_lab_store") and self._lab_store:
                    from cliver.gateway.routes.admin_labs import get_lab_routes

                    routes.extend(get_lab_routes(self._lab_store, admin_ctx, shared_auth))
                    logger.info("Lab routes registered")
            except Exception as e:
                logger.error(f"Failed to register lab routes: {e}")

            # MCP server routes (config.yaml backed)
            try:
                from cliver.gateway.routes.admin_mcp import get_mcp_routes

                routes.extend(get_mcp_routes(config_manager, shared_auth))
                logger.info("MCP server routes registered")
            except Exception as e:
                logger.error(f"Failed to register MCP server routes: {e}")

            routes.extend(admin_api_routes)
            # SPA catch-all appended LAST — after all API routes
            routes.extend(spa_routes)
            if admin_user and admin_pass:
                logger.info("Admin portal enabled at /admin")
            else:
                logger.info("Admin portal disabled (no credentials configured)")
        except Exception as e:
            logger.error(f"Failed to build admin routes: {e}")

        return routes

    def run(self) -> None:
        """Start the gateway as a uvicorn web server."""
        import uvicorn

        config = self._get_config_manager().config
        gw_config = config.gateway
        host = gw_config.host if gw_config else "127.0.0.1"
        port = gw_config.port if gw_config else 8321
        # Filter noisy polling endpoints from access logs
        logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())

        uvicorn.run(self.create_app(), host=host, port=port, log_level="info")

    async def _on_startup(self) -> None:
        """Lifecycle hook: acquire flock, init components."""
        logger.info("Gateway starting")
        self._start_time = time.monotonic()

        # Singleton flock
        self._pid_file = open(self._pid_path, "w")
        try:
            fcntl.flock(self._pid_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._pid_file.close()
            self._pid_file = None
            raise RuntimeError("Another gateway is already running") from None
        self._pid_file.write(str(os.getpid()))
        self._pid_file.flush()

        # Shared resources are initialized in init() before fork

        # Scheduler (APScheduler)
        try:
            db_path = self._agent_profile.db_path
            self._run_store = TaskStore(db_path)
            self._task_manager = TaskManager(self._agent_profile.tasks_dir, self._run_store)

            self._scheduler = Scheduler(
                task_manager=self._task_manager,
                run_store=self._run_store,
                run_task_fn=self._run_task,
            )
            if self._scheduler:
                self._scheduler.validate_tasks()
                self._scheduler.cleanup_orphan_runs()
                self._scheduler.sync_tasks()
                self._scheduler.start()
        except Exception as e:
            logger.error("Failed to initialize scheduler: %s", e)

        # Global asyncio exception handler — catches unhandled exceptions in tasks
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._asyncio_exception_handler)

        # Background cleanup loop
        asyncio.create_task(self._cleanup_loop(), name="cleanup-loop")

        # Adapters
        try:
            adapters = self._load_adapters()
            self._adapter_manager = AdapterManager(
                adapters,
                on_message=self._handle_message,
                on_reconnect=self._on_adapter_reconnect,
            )
            await self._adapter_manager.run()
        except Exception as e:
            logger.error(f"Failed to start adapters: {e}")

        # Session cleanup (reuses the store initialized in init())
        try:
            self._session_manager = self._cli_session_manager
            if self._session_manager:
                sc = self._get_config_manager().config.session
                deleted = self._session_manager.delete_stale_sessions(max_age_days=sc.max_age_days)
                if deleted:
                    logger.info("Cleaned up %d stale sessions (>%d days)", deleted, sc.max_age_days)
                deleted = self._session_manager.delete_oldest_sessions(keep=sc.max_sessions)
                if deleted:
                    logger.info("Cleaned up %d oldest sessions (kept %d)", deleted, sc.max_sessions)
        except Exception as e:
            logger.warning("Session cleanup failed: %s", e)

        logger.info("Gateway started")

    async def _on_cleanup(self) -> None:
        """Lifecycle hook: stop adapters, close DB, release flock."""
        logger.info("Gateway stopping")

        if self._scheduler:
            self._scheduler.shutdown(wait=False)

        if self._adapter_manager:
            try:
                await self._adapter_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping adapters: {e}")

        if self._run_store:
            try:
                self._run_store.close()
            except Exception as e:
                logger.error(f"Error closing run store: {e}")

        self._thread_queue.cleanup(max_idle_seconds=0)

        if self._pid_file:
            self._pid_file.close()
            self._pid_file = None
        if self._pid_path.exists():
            self._pid_path.unlink(missing_ok=True)

        logger.info("Gateway stopped")

    def _get_status(self) -> dict:
        """Return current gateway status for /health endpoint."""
        uptime = int(time.monotonic() - self._start_time) if self._start_time else 0
        platforms = self._adapter_manager.connected_platforms if self._adapter_manager else []
        adapter_statuses = self._adapter_manager.platform_statuses if self._adapter_manager else []
        return {
            "uptime": uptime,
            "tasks_run": self._tasks_run,
            "platforms": platforms,
            "adapters": adapter_statuses,
        }

    async def _get_status_async(self) -> dict:
        """Async wrapper for _get_status (used by admin API)."""
        return self._get_status()

    async def _cleanup_loop(self) -> None:
        """Background task: periodic cleanup and task resync."""
        while True:
            try:
                await asyncio.sleep(600)
                self._thread_queue.cleanup(max_idle_seconds=3600)
                if self._scheduler:
                    self._scheduler.sync_tasks()
            except asyncio.CancelledError:
                return

    # -- Task execution ----------------------------------------------------------

    async def _run_task(self, task: TaskDefinition) -> None:
        """Execute a task — origin-aware with reply-back for IM tasks."""
        execution_id = str(uuid.uuid4())[:8]
        logger.info(f"Running task '{task.name}' (execution: {execution_id})")

        # Load origin from database (not YAML)
        if not task.origin and self._run_store:
            task.origin = self._run_store.get_origin(task.name)

        if self._run_store:
            self._run_store.set_task_state(task.name, "running")

        # Check adapter availability for IM-origin tasks
        if task.origin and task.origin.platform:
            connected = self._adapter_manager.connected_platforms if self._adapter_manager else []
            if task.origin.platform not in connected:
                logger.warning(
                    "Task '%s' suspended: adapter '%s' not connected",
                    task.name,
                    task.origin.platform,
                )
                if self._run_store:
                    self._run_store.set_task_state(
                        task.name,
                        "suspended",
                        reason=f"Adapter '{task.origin.platform}' not connected",
                    )
                return

        run_record = TaskRun(
            task_name=task.name,
            execution_id=execution_id,
            status="running",
            started_at=TaskManager.timestamp_now(),
        )

        # Load session_id from DB if not already on the task
        if not task.session_id and self._run_store:
            task.session_id = self._run_store.get_session_id(task.name)

        try:
            conversation_history = None
            if task.session_id and self._session_manager:
                conversation_history = self._load_session_history(task.session_id)

            from cliver.skill_manager import SkillManager

            if task.skills:
                manager = SkillManager()
                for name in task.skills:
                    if not manager.get_skill(name):
                        logger.warning("Skill '%s' not found for pre-activation", name)

            agent = self._get_agent(task.model)
            response = await agent.chat(
                prompt=task.prompt,
                conversation=conversation_history,
                system_prompt=self._build_agent_system_prompt(agent),
            )

            response_text = response.message.text or "No response."

            run_record.status = "completed"
            run_record.result = response_text

            # Create a session for every execution so it's viewable in the admin
            if self._session_manager:
                session_id = self._session_manager.create_session(
                    title=f"{task.name} [{execution_id}]",
                )
                self._session_manager.append_turn(session_id, "user", task.prompt)
                self._session_manager.append_turn(
                    session_id, "assistant", response_text, message=response.message or None
                )
                run_record.session_id = session_id

            # Deliver result to IM origin, or save JSON for non-IM tasks
            if task.origin and task.origin.platform and task.origin.channel_id:
                await self._deliver_to_origin(task, response_text)
            else:
                self._save_result_json(task.name, run_record)

        except Exception as e:
            run_record.status = "failed"
            run_record.error = str(e)
            response_text = f"Task '{task.name}' failed: {e}"
            logger.error(f"Task '{task.name}' failed: {e}")

            # Create a session for failed executions too
            if self._session_manager:
                session_id = self._session_manager.create_session(
                    title=f"{task.name} [{execution_id}]",
                )
                self._session_manager.append_turn(session_id, "user", task.prompt)
                msg = CLIverMessage(role="assistant", content=response_text)
                self._session_manager.append_turn(
                    session_id,
                    "assistant",
                    response_text,
                    message=msg,
                )
                run_record.session_id = session_id

            if task.origin and task.origin.platform and task.origin.channel_id:
                try:
                    await self._deliver_to_origin(task, response_text)
                except Exception:
                    logger.error("Failed to deliver error to origin")

        finally:
            run_record.finished_at = TaskManager.timestamp_now()
            if self._run_store:
                self._run_store.record_run(run_record)
                self._run_store.set_task_state(task.name, run_record.status)

            # Clear run_at after one-shot execution
            if task.run_at and self._task_manager:
                task.run_at = None
                self._task_manager.save_task(task)
                logger.info(f"Cleared run_at for one-shot task '{task.name}'")
                if self._scheduler:
                    self._scheduler.sync_tasks()

    def _load_session_history(self, session_id: str):
        """Load conversation history from a linked session for task context."""
        turns = self._session_manager.load_turns(session_id)
        if not turns:
            return None

        return [CLIverMessage(role=turn["role"], content=turn["content"]) for turn in turns] or None

    async def _deliver_to_origin(self, task: TaskDefinition, response_text: str) -> None:
        """Deliver task result to the originating IM thread and append synthetic turns."""
        adapter = self._get_adapter(task.origin.platform)
        if not adapter:
            logger.error(f"No adapter for platform '{task.origin.platform}'")
            return

        formatted = adapter.format_message(response_text)
        chunks = split_message(formatted, adapter.max_message_length())
        for chunk in chunks:
            await adapter.send_text(
                task.origin.channel_id,
                chunk,
                reply_to=task.origin.thread_id,
            )
        logger.info(
            "Task '%s' result delivered to %s (channel=%s, thread=%s, %d chunk(s))",
            task.name,
            task.origin.platform,
            task.origin.channel_id,
            task.origin.thread_id,
            len(chunks),
        )

        # Append synthetic turns to the linked session
        if self._session_manager and task.session_id:
            self._session_manager.append_turn(
                task.session_id,
                "user",
                f"[Task '{task.name}' executed]",
            )
            self._session_manager.append_turn(
                task.session_id,
                "assistant",
                response_text,
            )
            self._session_manager.save_options(
                task.session_id,
                {
                    "task_origin": task.origin.model_dump(exclude_none=True),
                    "task_name": task.name,
                },
            )

    def _save_result_json(self, task_name: str, run_record: TaskRun) -> None:
        """Save execution result as a JSON file for non-IM tasks."""
        import json

        task_results_dir = self._agent_profile.tasks_dir / task_name
        task_results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task_name}_execution_{run_record.execution_id}.json"
        result_path = task_results_dir / filename

        result_data = {
            "task_name": run_record.task_name,
            "execution_id": run_record.execution_id,
            "status": run_record.status,
            "started_at": run_record.started_at,
            "finished_at": run_record.finished_at,
            "result": run_record.result,
            "error": run_record.error,
        }

        result_path.write_text(
            json.dumps(result_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Task result saved to %s", result_path)

    def _init_shared_resources(self):
        """Initialize shared MCP client, builtin tools, agent profile."""
        config_manager = self._get_config_manager()

        from cliver.util import configure_timezone

        configure_timezone(config_manager.config.timezone)

        from cliver.agent_profile import set_current_profile

        self._agent_profile = CliverProfile(self.config_dir)
        set_current_profile(self._agent_profile)
        self._agent_profile.ensure_dirs()

        self._builtin_tools = discover_builtin_tools()
        reg = ToolRegistry(self._builtin_tools)
        reg.configure(config_manager.config.enabled_toolsets)
        self._builtin_tools = reg.all_tools

        self._mcp_client = MCPClient(config_manager.list_mcp_servers_for_mcp_caller())

    def _build_agent_system_prompt(self, agent: Agent, *, extra: str | None = None) -> str | None:
        """Return extra system prompt content for this agent call.

        The builtin prompt (models, tools, self-awareness) is generated
        automatically by AgentCore — this method only returns runtime context.
        """
        return extra

    def _get_agent(self, model_name: str | None = None) -> Agent:
        """Get or create an Agent for the given model."""
        from cliver.agent_factory import create_agent_core

        config_manager = self._get_config_manager()
        model_name = model_name or config_manager.get_llm_model().name
        if model_name in self._agents:
            return self._agents[model_name]

        mc = self._resolve_model(model_name)
        if not mc:
            raise ValueError(f"Model '{model_name}' not found in config.")

        agent_core = create_agent_core(
            model_config=mc,
            config_manager=config_manager,
            on_event=_create_gateway_tool_handler(),
        )

        agent = Agent(name="gateway", config=self._get_default_agent_config(), agent_core=agent_core)
        self._agents[model_name] = agent
        return agent

    def _resolve_model(self, name: str):
        config_manager = self._get_config_manager()
        models = config_manager.list_llm_models()
        if name in models:
            return models[name]
        suffix = f"/{name}"
        matches = [mc for key, mc in models.items() if key.endswith(suffix)]
        return matches[0] if len(matches) == 1 else None

    def _get_default_model_name(self) -> str | None:
        config_manager = self._get_config_manager()
        mc = config_manager.get_llm_model()
        return mc.name if mc else None

    def _get_default_agent_config(self):
        from cliver.config import AgentConfig

        return AgentConfig(name="gateway-agent")

    # -- Session & adapter resolution -----------------------------------------

    @staticmethod
    def _resolve_session_key(event: MessageEvent) -> str:
        """Build a session key from a message event.

        Each thread gets its own session. For top-level messages (no thread),
        the message's own ID becomes the thread root — the bot's reply will
        create a new thread, giving each conversation its own context.

        Key format: platform:channel_id:thread_id
        """
        thread = event.thread_id or event.message_id or ""
        return f"{event.platform}:{event.channel_id}:{thread}"

    @staticmethod
    def _resolve_adapter_class(adapter_type: str) -> str:
        """Resolve an adapter type to its full module.ClassName path.

        Builtin types (no dots) are looked up in BUILTIN_ADAPTERS.
        Custom types (with dots) are returned as-is.
        """
        if "." not in adapter_type:
            if adapter_type not in BUILTIN_ADAPTERS:
                raise ValueError(
                    f"Unknown adapter type '{adapter_type}'. Available: {', '.join(BUILTIN_ADAPTERS.keys())}"
                )
            return BUILTIN_ADAPTERS[adapter_type]
        return adapter_type

    @staticmethod
    def _import_adapter_class(class_path: str):
        """Dynamically import and return an adapter class."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _load_adapters(self) -> List[PlatformAdapter]:
        """Load and instantiate adapters from config."""
        config_manager = self._get_config_manager()
        gateway_config = config_manager.config.gateway
        if not gateway_config or not gateway_config.platforms:
            return []

        adapters = []
        for name, platform_config in gateway_config.platforms.items():
            try:
                if not platform_config.name:
                    platform_config.name = name
                class_path = self._resolve_adapter_class(platform_config.type)
                adapter_cls = self._import_adapter_class(class_path)
                adapter = adapter_cls(platform_config)
                adapters.append(adapter)
                logger.info(f"Loaded adapter: {adapter.name}")
            except Exception as e:
                logger.error(f"Failed to load adapter '{name}': {e}")

        return adapters

    # -- Message handling -----------------------------------------------------

    async def _compress_history(self, history, new_input: str):
        """Compress conversation history — skipped in v1 (needs porting)."""
        return history

    def _asyncio_exception_handler(self, loop, context):
        exception = context.get("exception")
        message = context.get("message", "")
        logger.error("Asyncio unhandled exception: %s (message: %s)", exception, message)
        if exception:
            import traceback

            logger.error("".join(traceback.format_exception(type(exception), exception, exception.__traceback__)))

    async def _handle_message(self, event: MessageEvent) -> None:
        """Central message handler -- routes platform messages to AgentCore.

        Each message is dispatched as a separate asyncio.Task to ensure
        ContextVar isolation (im_context) between concurrent conversations.
        """
        asyncio.create_task(
            self._handle_message_safe(event),
            name=f"msg:{event.platform}:{event.channel_id}",
        )

    async def _handle_message_safe(self, event: MessageEvent) -> None:
        try:
            await self._handle_message_inner(event)
        except Exception as e:
            logger.exception("_handle_message crashed: %s", e)

    async def _handle_message_inner(self, event: MessageEvent) -> None:
        logger.info(
            "_handle_message called: platform=%s, user=%s, text=%.100s", event.platform, event.user_id, event.text
        )

        # Find the adapter for this platform
        adapter = self._get_adapter(event.platform)
        if not adapter:
            logger.error(f"No adapter found for platform '{event.platform}'")
            return

        # Resolve session — uses the shared SessionManager instance
        session_key = self._resolve_session_key(event)
        async with self._thread_queue.get_lock(session_key):
            sm = self._session_manager

            # Get or create session
            session_id = self._get_or_create_session(sm, session_key)

            # Show typing indicator
            try:
                await adapter.send_typing(event.channel_id)
            except Exception as e:
                logger.debug(f"Typing indicator failed: {e}")

            # Prepare media as temp files for AgentCore
            images, audio_files = [], []
            transcribed_texts = []
            for media in event.media:
                if media.data:
                    suffix = _media_suffix(media)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(media.data)
                    tmp.close()
                    if media.type == "image":
                        images.append(tmp.name)
                    elif media.type == "voice":
                        # Auto-transcribe voice messages to text
                        try:
                            from cliver.tools.transcribe_audio import transcribe_voice_message

                            transcript = await transcribe_voice_message(tmp.name)
                            if transcript:
                                transcribed_texts.append(transcript)
                                logger.info("Voice message transcribed: %.100s", transcript)
                            else:
                                audio_files.append(tmp.name)
                        except Exception as e:
                            logger.debug(f"Voice transcription failed, passing raw audio: {e}")
                            audio_files.append(tmp.name)

            # Prepend transcribed voice text to user input
            if transcribed_texts:
                voice_text = " ".join(transcribed_texts)
                if event.text:
                    event.text = f"{event.text}\n\n[Voice message]: {voice_text}"
                else:
                    event.text = voice_text

            # Load conversation history
            turns = sm.load_turns(session_id)
            conversation_history = [CLIverMessage(role=turn["role"], content=turn["content"]) for turn in turns]

            # Compress history if it exceeds the model's context window
            if conversation_history:
                try:
                    conversation_history = await self._compress_history(conversation_history, event.text)
                except Exception as e:
                    logger.warning("History compression failed, using full history: %s", e)

            # Check if this session is linked to a task
            session_opts = sm.load_options(session_id)
            linked_task_name = session_opts.get("task_name")
            linked_origin = session_opts.get("task_origin")

            # Record user turn
            sm.append_turn(
                session_id,
                "user",
                event.text,
                message=CLIverMessage(role="user", content=event.text),
            )

            # Run Agent
            default_model = self._get_default_model_name()
            logger.info(
                "Calling Agent.chat (model=%s, history=%d turns, task=%s)",
                default_model,
                len(conversation_history),
                linked_task_name or "none",
            )
            # Set IM context for tools (e.g., create_task) to read.
            im_ctx = {
                "platform": linked_origin["platform"] if linked_origin else event.platform,
                "channel_id": linked_origin.get("channel_id") if linked_origin else event.channel_id,
                "thread_id": linked_origin.get("thread_id") if linked_origin else (event.thread_id or event.message_id),
                "user_id": event.user_id,
                "session_id": session_id,
            }
            if linked_task_name:
                im_ctx["task_name"] = linked_task_name
            im_context.set(im_ctx)

            _reply_to_parts = [im_ctx["platform"], im_ctx.get("channel_id", "")]
            if im_ctx.get("thread_id"):
                _reply_to_parts.append(im_ctx["thread_id"])
            _reply_to_token = ":".join(_reply_to_parts)

            im_context_text = (
                "# IM Context\n\n"
                "You are responding in an IM conversation (e.g. Slack, Telegram).\n\n"
                "When you need to ask the user a question, use the structured format "
                "described in the Interaction Guidelines.\n\n"
                "## Task Creation Rules\n\n"
                "Prefer the CreateTask tool.\n"
                f"If you must use the shell command instead, include `--reply-to '{_reply_to_token}'` "
                "so results are delivered back to this conversation.\n"
            )
            if linked_task_name:
                im_context_text += f"\nYou are continuing from task '{linked_task_name}'. Reply in this same thread."

            try:
                agent = self._get_agent(default_model)
                response = await agent.chat(
                    prompt=event.text,
                    conversation=conversation_history or None,
                    system_prompt=self._build_agent_system_prompt(agent, extra=im_context_text),
                )
                response_text = response.message.text or "No response."
                logger.info("Agent response: %.200s", response_text)

                # Sync any tasks created via CreateTask tool during this chat
                if self._scheduler:
                    self._scheduler.sync_tasks()
            except Exception as e:
                logger.exception("Agent error: %s", e)
                response_text = f"Error: {e}"
            finally:
                im_context.set(None)

            # Record assistant turn and trim if session is getting large
            sm.append_turn(session_id, "assistant", response_text, message=response.message or None)
            sc = self._get_config_manager().config.session
            trimmed = sm.trim_turns(session_id, keep_last=sc.max_turns_per_session)
            if trimmed:
                logger.info("Trimmed %d old turns from session %s", trimmed, session_id)

            # Reply in-thread: use existing thread_id, or message_id to create a new thread
            reply_to = event.thread_id or event.message_id

            # Strip image URLs from text when media attachments are present
            if response.media:
                import re

                response_text = re.sub(r"https?://\S+\.(png|jpg|jpeg|gif|webp|svg)\S*", "", response_text)
                response_text = re.sub(r"Generated \d+ image\(s\):\s*", "", response_text)
                response_text = response_text.strip()

            # Send text response
            formatted = adapter.format_message(response_text) if response_text else ""
            chunks = split_message(formatted, adapter.max_message_length()) if formatted else []
            logger.info(
                "Sending %d chunk(s) to %s channel %s (thread: %s)",
                len(chunks),
                event.platform,
                event.channel_id,
                reply_to,
            )

            for chunk in chunks:
                try:
                    await adapter.send_text(event.channel_id, chunk, reply_to=reply_to)
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")

            # Send media (images, files, audio)
            if response.media:
                for media in response.media:
                    try:
                        data = media.to_bytes() if media.data else None
                        if not data:
                            continue
                        if media.type.value == "image":
                            await adapter.send_image(
                                event.channel_id, data, caption=media.filename or "", reply_to=reply_to
                            )
                        elif media.type.value == "audio":
                            await adapter.send_voice(event.channel_id, data, reply_to=reply_to)
                        else:
                            await adapter.send_file(
                                event.channel_id,
                                data,
                                filename=media.filename or f"file{media.get_file_extension()}",
                                reply_to=reply_to,
                            )
                        logger.info("Sent %s media to %s", media.type.value, event.channel_id)
                    except Exception as e:
                        logger.error("Failed to send %s media: %s", media.type.value, e)

    def _get_adapter(self, platform_name: str) -> Optional[PlatformAdapter]:
        """Find the adapter for a given platform name."""
        if not self._adapter_manager:
            return None
        for adapter in self._adapter_manager._adapters:
            if adapter.name == platform_name:
                return adapter
        return None

    def _get_or_create_session(self, sm: SessionManager, session_key: str) -> str:
        """Get existing session by key or create a new one.

        Uses the session title field to store the session key for lookup.
        """
        sessions = sm.list_sessions()
        for s in sessions:
            if s.get("title") == session_key:
                return s["id"]
        return sm.create_session(title=session_key)

    async def _on_adapter_reconnect(self, platform_name: str) -> None:
        """Resume suspended tasks when an adapter reconnects."""
        if not self._run_store:
            return

        task_manager = TaskManager(self._agent_profile.tasks_dir, self._run_store)

        suspended = self._run_store.get_tasks_by_status("suspended")
        for state in suspended:
            task = task_manager.get_task(state["task_name"])
            if task and task.origin and task.origin.platform == platform_name:
                self._run_store.set_task_state(task.name, "pending")
                logger.info(
                    "Resumed suspended task '%s' (adapter '%s' reconnected)",
                    task.name,
                    platform_name,
                )


class _QuietPollFilter(logging.Filter):
    """Suppress access log entries for high-frequency polling endpoints."""

    _QUIET_PATHS = ("/admin/api/status", "/health")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET_PATHS)


def _looks_like_base64(s: str) -> bool:
    """Quick heuristic: long string of alphanumeric + /+=."""
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9+/=\s]{100,}", s[:200]))


def _create_gateway_tool_handler():
    """Create a tool event handler that logs via the standard logger."""
    from cliver.events import ToolEvent, ToolEventType

    tool_logger = logging.getLogger("cliver.gateway.tools")

    async def _handler(event: ToolEvent) -> None:
        if event.event == ToolEventType.START:
            args_summary = ""
            if event.args:
                parts = []
                for k, v in event.args.items():
                    val = str(v)
                    if len(val) > 200:
                        val = val[:197] + "…"
                    parts.append(f"{k}={val}")
                args_summary = " " + ", ".join(parts[:5])
            tool_logger.info("[START] %s%s", event.tool_name, args_summary)

        elif event.event == ToolEventType.END:
            duration = f" ({event.duration_ms:.0f}ms)" if event.duration_ms else ""
            tool_logger.info("[DONE]  %s%s", event.tool_name, duration)
            if event.result:
                for line in event.result.splitlines():
                    if len(line) > 500 and not line[:20].isascii():
                        tool_logger.info("        [binary data, %d bytes]", len(line))
                    elif len(line) > 500 and _looks_like_base64(line.strip()):
                        tool_logger.info("        [base64 data, ~%dKB]", len(line) * 3 // 4 // 1024)
                    else:
                        tool_logger.info("        %s", line[:500])

        elif event.event == ToolEventType.ERROR:
            duration = f" ({event.duration_ms:.0f}ms)" if event.duration_ms else ""
            tool_logger.warning("[ERROR] %s%s: %s", event.tool_name, duration, event.error)

    return _handler


def _media_suffix(media: MediaAttachment) -> str:
    """Determine file suffix from media type/mime."""
    if media.mime_type:
        ext = media.mime_type.split("/")[-1]
        return f".{ext}"
    if media.type == "image":
        return ".png"
    if media.type == "voice":
        return ".ogg"
    return ".bin"
