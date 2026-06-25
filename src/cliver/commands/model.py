import click
from rich import box
from rich.panel import Panel
from rich.table import Table

from cliver.cli import Cliver, pass_cliver
from cliver.commands import click_help, wants_help
from cliver.config import ModelConfig
from cliver.util import parse_key_value_options


@click.group(name="model", help="Manage LLM model configurations (list, add, update, set default, remove)")
@click.pass_context
def model(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


# ---------------------------------------------------------------------------
# Business logic functions
# ---------------------------------------------------------------------------


def _resolve_model(config_manager, name: str) -> tuple[ModelConfig | None, str | None]:
    """Resolve a model by canonical name (provider/model) or short name.

    Returns (ModelConfig, provider_name) or (None, None).
    """
    models = config_manager.list_llm_models()
    if name in models:
        return models[name], models[name].provider
    for key, mc in models.items():
        if key.endswith(f"/{name}"):
            return mc, mc.provider
    return None, None


def _list_models(cliver: Cliver):
    """List all configured LLM models grouped by provider."""
    config_manager = cliver.config_manager
    models = config_manager.list_llm_models()
    if not models:
        cliver.output("No LLM Models configured.")
        return

    default_name = config_manager.config.default_model

    table = Table(title="Configured LLM Models", box=box.SQUARE)
    table.add_column("", min_width=1, max_width=1, no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("API Model", style="yellow")
    table.add_column("Provider", style="cyan")
    table.add_column("Type", style="blue")

    for name, mc in models.items():
        modality = mc.category or "text"
        api_model = mc.model

        marker = "✔" if name == default_name else ""
        table.add_row(marker, name, api_model, mc.provider, modality)

    cliver.output(table)


def _show_model_detail(cliver: Cliver, name: str):
    """Show detailed information about a specific model."""
    config_manager = cliver.config_manager
    mc, provider_name = _resolve_model(config_manager, name)
    if not mc:
        cliver.output(f"Model '{name}' not found.")
        models = config_manager.list_llm_models()
        if models:
            cliver.output(f"Available: {', '.join(models.keys())}")
        return

    is_default = config_manager.config.default_model == mc.name
    provider_config = config_manager.list_providers().get(mc.provider)

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("Key", style="dim", min_width=18)
    t.add_column("Value")

    t.add_row("Name", mc.name)
    if is_default:
        t.add_row("Status", "default")
    t.add_row("Provider", mc.provider)

    if provider_config:
        t.add_row("URL", provider_config.api_url)
        if provider_config.api_key:
            t.add_row("API Key", f"(from provider) {_mask_api_key(provider_config.api_key)}")

    t.add_row("Category", mc.category or "text")
    t.add_row("API Model", mc.model)
    features = _format_features(mc)
    if features != "-":
        t.add_row("Features", features)

    if mc.options:
        for k, v in mc.options.model_dump().items():
            t.add_row(f"  {k}", str(v))

    cliver.output(Panel(t, title=f"Model: {mc.name}", border_style="green", padding=(0, 1)))


def _mask_api_key(key: str) -> str:
    """Mask API key — show template expressions, mask plain text."""
    if "{{" in key and "}}" in key:
        return f"{key}"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def _format_features(mc) -> str:
    """Format model features as tags based on properties."""
    features = []
    if mc.is_text:
        features.append("tools")
    if mc.can_strict_tool_call:
        features.append("fn-call")
    return " ".join(features) if features else "-"


class ModelNameType(click.ParamType):
    """Click parameter type that provides shell completion for model names."""

    name = "model_name"

    def shell_complete(self, ctx, param, incomplete):
        try:
            cliver_obj = ctx.find_object(Cliver)
            if cliver_obj:
                # Only suggest text models — non-text models can't be default
                models = cliver_obj.config_manager.list_text_models()
                return [click.shell_completion.CompletionItem(n) for n in models if n.startswith(incomplete)]
        except Exception:
            pass
        return []


def _set_default_model(cliver: Cliver, name: str = None):
    """Set or show the default LLM model (text models only)."""
    config_manager = cliver.config_manager
    text_models = config_manager.list_text_models()

    if not name:
        default = config_manager.get_llm_model()
        if default:
            cliver.output(f"Current default model: {default.name}")
        else:
            cliver.output("No default model set.")

        if text_models:
            cliver.output(f"Available models: {', '.join(text_models.keys())}")
        return

    # Validate that the model exists and is a text model
    mc = text_models.get(name)
    if mc is None:
        # Try resolving by suffix across all models to give a precise error
        resolved, _ = _resolve_model(config_manager, name)
        if resolved and resolved.category != "text":
            cliver.output(f"Model '{name}' is a '{resolved.category}' model. Only text models can be set as default.")
        else:
            cliver.output(f"Model '{name}' not found.")
            if text_models:
                cliver.output(f"Available text models: {', '.join(text_models.keys())}")
        return

    if config_manager.set_default_model(name):
        cliver.output(f"Default model set to: {name}")
    else:
        cliver.output(f"Failed to set default model to '{name}'.")


def _remove_model(cliver: Cliver, name: str):
    """Remove a LLM model."""
    config_manager = cliver.config_manager
    mc, _ = _resolve_model(config_manager, name)
    if not mc:
        cliver.output(f"No LLM Model found with name: {name}")
        return
    canonical = mc.name
    config_manager.remove_llm_model(name)
    cliver.output(f"Removed LLM Model: {canonical}")


def _add_model(
    cliver: Cliver,
    provider: str,
    model_name: str,
    api_model: str = "",
    option: tuple = None,
):
    """Add a new LLM model to a provider."""
    config_manager = cliver.config_manager

    if provider not in config_manager.list_providers():
        cliver.output(f"Provider '{provider}' not found.")
        return

    models = config_manager.list_llm_models()
    if model_name in models and models[model_name].provider == provider:
        cliver.output(f"Model '{model_name}' already exists under provider '{provider}'.")
        return

    is_first = not config_manager.list_llm_models()

    options_dict = None
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    config_manager.add_or_update_llm_model(
        provider,
        model_name,
        api_model_name=api_model,
        options=options_dict,
    )

    # Resolve API model name (may differ from CLIver name)
    mc, _ = _resolve_model(config_manager, model_name)
    api_model = mc.model if mc and mc.model != model_name else None

    if is_first:
        suffix = f" (API: {api_model})" if api_model else ""
        cliver.output(f"Added LLM Model: {model_name}{suffix} (set as default)")
    else:
        cliver.output(f"Added LLM Model: {model_name}" + (f" (API: {api_model})" if api_model else ""))


def _update_model(
    cliver: Cliver,
    name: str,
    api_model: str = "",
    option: tuple = None,
):
    """Update an existing LLM model."""
    config_manager = cliver.config_manager
    mc, provider_name = _resolve_model(config_manager, name)
    if not mc:
        cliver.output(f"LLM Model '{name}' not found.")
        return

    options_dict = None
    if option:
        options_dict = parse_key_value_options(option, cliver.console)

    model_key = mc.name.split("/", 1)[1] if "/" in mc.name else mc.name
    config_manager.add_or_update_llm_model(
        provider_name,
        model_key,
        api_model_name=api_model,
        options=options_dict,
    )
    cliver.output(f"LLM Model: {mc.name} updated" + (f" (API: {api_model})" if api_model else ""))


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

_SUBCOMMANDS: dict[str, click.Command] = {}


def dispatch(cliver: Cliver, args: str):
    """Manage LLM models — list, add, set, remove, default."""
    parts = args.strip().split(None, 1) if args.strip() else []
    sub = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("--help", "-h", "help"):
        cliver.output(click_help(model, "/model"))
        return

    if sub in _SUBCOMMANDS and wants_help(rest):
        cliver.output(click_help(_SUBCOMMANDS[sub], f"/model {sub}"))
        return

    if sub == "list":
        _list_models(cliver)
    elif sub == "show":
        model_name = rest.strip()
        if not model_name:
            cliver.output(click_help(_SUBCOMMANDS["show"], "/model show"))
            return
        _show_model_detail(cliver, model_name)
    elif sub == "default":
        model_name = rest.strip() if rest else None
        _set_default_model(cliver, model_name)
    elif sub == "add":
        _dispatch_add(cliver, rest)
    elif sub == "set":
        _dispatch_set(cliver, rest)
    elif sub == "remove":
        name = rest.strip()
        if not name:
            cliver.output(click_help(_SUBCOMMANDS["remove"], "/model remove"))
            return
        _remove_model(cliver, name)
    else:
        _show_model_detail(cliver, sub)


def _parse_model_flags(rest: str) -> dict:
    """Parse --flag value pairs from a rest string."""
    from shlex import split as shlex_split

    try:
        tokens = shlex_split(rest)
    except ValueError:
        tokens = rest.split()

    opts: dict = {}
    options = []
    i = 0
    flag_map = {
        "--name": "name",
        "-n": "name",
        "--model": "model",
        "-m": "model",
        "--provider": "provider",
        "-p": "provider",
    }
    while i < len(tokens):
        if tokens[i] in flag_map and i + 1 < len(tokens):
            opts[flag_map[tokens[i]]] = tokens[i + 1]
            i += 2
        elif tokens[i] in ("--option", "-o") and i + 1 < len(tokens):
            options.append(tokens[i + 1])
            i += 2
        else:
            i += 1
    if options:
        opts["option"] = tuple(options)
    return opts


def _dispatch_add(cliver: Cliver, rest: str) -> None:
    """Handle /model add from TUI dispatch."""
    opts = _parse_model_flags(rest)
    name = opts.get("name")
    provider = opts.get("provider")
    if not name or not provider:
        cliver.output("Usage: /model add --provider NAME --name MODEL_NAME [--option K=V]")
        return
    _add_model(cliver, provider, name, api_model=opts.get("model", ""), option=opts.get("option"))


def _dispatch_set(cliver: Cliver, rest: str) -> None:
    """Handle /model set from TUI dispatch."""
    opts = _parse_model_flags(rest)
    name = opts.get("name")
    if not name:
        cliver.output("Usage: /model set --name NAME [--option K=V]")
        return
    _update_model(cliver, name, api_model=opts.get("model", ""), option=opts.get("option"))


# ---------------------------------------------------------------------------
# Click commands (thin wrappers)
# ---------------------------------------------------------------------------


# noinspection PyUnresolvedReferences
@model.command(name="list", help="List all configured models with provider, modalities, and active status")
@pass_cliver
def list_llm_models(cliver: Cliver):
    _list_models(cliver)


# noinspection PyUnresolvedReferences
@model.command(name="default", help="Show or set the default LLM model for new sessions")
@click.argument("name", type=ModelNameType(), required=False)
@pass_cliver
def set_default_model(cliver: Cliver, name: str):
    """Set or show the default LLM model."""
    _set_default_model(cliver, name)


# noinspection PyUnresolvedReferences
@model.command(name="remove", help="Remove a model configuration permanently by name")
@click.argument("name", type=str)
@pass_cliver
def remove_llm_model(cliver: Cliver, name: str):
    _remove_model(cliver, name)


# noinspection PyUnresolvedReferences
@model.command(name="add", help="Register a new model under an existing provider")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="CLIver name for this model (e.g. 'my-deepseek')",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="",
    help="API-facing model name sent to the provider (defaults to --name if not set)",
)
@click.option(
    "--provider",
    "-p",
    type=str,
    required=True,
    help="Provider name (e.g. 'deepseek')",
)
@click.option(
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Model option as key=value (repeatable, e.g. -o temperature=0.7)",
)
@pass_cliver
def add_llm_model(cliver: Cliver, name: str, model: str, provider: str, option: tuple):
    _add_model(cliver, provider, name, api_model=model, option=option)


# noinspection PyUnresolvedReferences
@model.command(name="set", help="Update options of an existing model")
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Model name in canonical (provider/model) or short form",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="",
    help="Change the API-facing model name sent to the provider",
)
@click.option(
    "--option",
    "-o",
    multiple=True,
    type=str,
    help="Option as key=value to update (repeatable, e.g. -o temperature=0.5)",
)
@pass_cliver
def update_llm_model(cliver: Cliver, name: str, model: str, option: tuple):
    _update_model(cliver, name, api_model=model, option=option)


@model.command(name="show", help="Show detailed information about a specific model")
@click.argument("name", type=str)
@pass_cliver
def show_llm_model(cliver: Cliver, name: str):
    _show_model_detail(cliver, name)


# Populate subcommand map for dispatch help generation.
_SUBCOMMANDS.update(
    {
        "list": list_llm_models,
        "show": show_llm_model,
        "default": set_default_model,
        "add": add_llm_model,
        "set": update_llm_model,
        "remove": remove_llm_model,
    }
)
