import { useState, useCallback, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  ThreadPrimitive,
  MessagePrimitive,
  type ThreadMessageLike,
  type AppendMessage,
} from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { ArrowUp, Plus, Square, X, Trash2 } from "lucide-react";
import { streamChat, type ChatArtifact } from "@/lib/chat-stream";
import { useConversation } from "@/hooks/use-conversations";
import { useAgents, useSkills, useTemplates } from "@/hooks/use-api";
import { ConversationSidebar } from "@/components/chat/ConversationSidebar";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { useTranslation } from "@/i18n";
import { apiDelete } from "@/lib/api";
import { SessionOptionsSchema } from "@/lib/schemas";

function generateId(): string {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

function convertTurnToMessage(turn: { role: string; content: string }): ThreadMessageLike {
  return {
    id: generateId(),
    role: turn.role as "user" | "assistant",
    content: [{ type: "text" as const, text: turn.content }],
  };
}

export default function ChatPage() {
  const { t } = useTranslation();
  const { conversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const activeConversationId = conversationId || null;

  // Messages stored per conversation — allows background streaming after switch
  const [messagesByConv, setMessagesByConv] = useState<Record<string, ThreadMessageLike[]>>({});
  const [runningConvId, setRunningConvId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastArtifacts, setLastArtifacts] = useState<ChatArtifact[]>([]);
  const [artifactMessageId, setArtifactMessageId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const loadedConversationIds = useRef<Set<string>>(new Set());
  const turnDbIdByMsgId = useRef<Record<string, number>>({});
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  const messages = activeConversationId
    ? (messagesByConv[activeConversationId] || [])
    : [];
  const isRunning = !!(runningConvId && runningConvId === activeConversationId);

  // Composer state
  const [inputText, setInputText] = useState("");
  const [showConfigMenu, setShowConfigMenu] = useState(false);

  const { data: conversationDetail } = useConversation(activeConversationId);
  const { data: templates } = useTemplates();

  // Per-chat config — local state synced with session options on load
  const [selectedAgent, setSelectedAgent] = useState("");
  const [systemMessage, setSystemMessage] = useState("");
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const lastLoadedSessionId = useRef<string | null>(null);

  // Load config from session options — runs only once per conversation load.
  // lastLoadedSessionId tracks which session's data was applied, so re-renders
  // from conversationDetail reference changes don't overwrite optimistic state.
  useEffect(() => {
    if (!activeConversationId) {
      setSelectedAgent("");
      setSystemMessage("");
      setSelectedSkills([]);
      lastLoadedSessionId.current = null;
      return;
    }
    const dataId = conversationDetail?.session?.id;
    if (!dataId || dataId !== activeConversationId) return;
    if (lastLoadedSessionId.current === dataId) return;
    lastLoadedSessionId.current = dataId;

    const opts = SessionOptionsSchema.parse(
      (conversationDetail?.session?.options as Record<string, unknown>) || {},
    );
    setSelectedAgent(String(opts.agent || ""));
    setSystemMessage(String(opts.system_prompt || ""));
    setSelectedSkills(opts.skills ?? []);
  }, [activeConversationId, conversationDetail]);

  // Clear load tracker when leaving a conversation so returning reloads
  useEffect(() => {
    return () => { lastLoadedSessionId.current = null; };
  }, [activeConversationId]);

  // Persist config changes — sends only the changed fields as a merge patch
  const persistPatch = useCallback(
    (patch: Record<string, unknown>) => {
      if (!activeConversationId) return;
      fetch(`/admin/api/conversations/${encodeURIComponent(activeConversationId)}`, {
        method: "PATCH",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ options: patch }),
      }).catch(() => {});
    },
    [activeConversationId],
  );

  const handleSetAgent = useCallback((v: string) => {
    setSelectedAgent(v);
    persistPatch({ agent: v || null });
  }, [persistPatch]);
  const handleSetSystemMessage = useCallback((v: string) => {
    setSystemMessage(v);
    persistPatch({ system_prompt: v || null });
  }, [persistPatch]);
  const handleSetSkills = useCallback((v: string[]) => {
    setSelectedSkills(v);
    persistPatch({ skills: v.length > 0 ? v : null });
  }, [persistPatch]);

  // Constrain App wrapper height so only the conversation viewport scrolls
  useEffect(() => {
    const el = document.getElementById("app-content");
    if (!el) return;
    const prevHeight = el.style.height;
    const prevOverflow = el.style.overflow;
    el.style.height = "100vh";
    el.style.overflow = "hidden";
    return () => {
      el.style.height = prevHeight;
      el.style.overflow = prevOverflow;
    };
  }, []);

  // Load turns when active conversation changes
  useEffect(() => {
    if (activeConversationId && runningConvId === activeConversationId) return;
    if (activeConversationId && conversationDetail?.turns) {
      const dataId = conversationDetail.session?.id;
      if (dataId && dataId !== activeConversationId) return;
      if (loadedConversationIds.current.has(activeConversationId)) return;
      loadedConversationIds.current.add(activeConversationId);
      const msgs = conversationDetail.turns.map((turn) => {
        const msg = convertTurnToMessage(turn);
        turnDbIdByMsgId.current[msg.id!] = turn.id;
        return msg;
      });
      setMessagesByConv((prev) => ({
        ...prev,
        [activeConversationId]: msgs,
      }));
    } else if (!activeConversationId) {
      setError(null);
    }
  }, [activeConversationId, conversationDetail, runningConvId]);

  // Scroll to bottom when messages change (history load or new messages)
  useEffect(() => {
    if (scrollAnchorRef.current) {
      scrollAnchorRef.current.scrollIntoView({ behavior: "instant" as ScrollBehavior });
    }
  }, [messages]);

  const handleSend = useCallback(() => {
    const text = inputText.trim();
    if (!text || isRunning) return;
    onNewRef.current?.({
      role: "user",
      content: [{ type: "text" as const, text }],
      parentId: null,
      sourceId: null,
      runConfig: undefined,
    } as unknown as AppendMessage);
    setInputText("");
  }, [inputText, isRunning]);

  const handleApplyTemplate = useCallback((tpl: { system_prompt: string; skills: string[]; agent?: string | null }) => {
    const vAgent = tpl.agent || "";
    const vSkills = tpl.skills;
    setSelectedAgent(vAgent);
    setSystemMessage(tpl.system_prompt);
    setSelectedSkills(vSkills);
    // Send all three as a single merge patch
    persistPatch({
      agent: vAgent || null,
      system_prompt: tpl.system_prompt || null,
      skills: vSkills.length > 0 ? vSkills : null,
    });
  }, [persistPatch]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      // Don't submit while IME is composing (e.g. Chinese/Japanese/Korean input)
      if (e.nativeEvent.isComposing) return;
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleNewChat = useCallback(() => {
    navigate("/admin/chat");
  }, [navigate]);

  const handleDelete = useCallback(
    (id: string) => {
      if (activeConversationId === id) {
        navigate("/admin/chat", { replace: true });
      }
      setMessagesByConv((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
    [activeConversationId, navigate, queryClient],
  );

  // -- Turn deletion --
  const [turnToDelete, setTurnToDelete] = useState<{ msgId: string } | null>(null);

  const handleDeleteTurn = useCallback(async () => {
    if (!turnToDelete || !activeConversationId) return;
    const dbTurnId = turnDbIdByMsgId.current[turnToDelete.msgId];
    if (dbTurnId == null) return;

    try {
      await apiDelete(
        `/conversations/${encodeURIComponent(activeConversationId)}/turns/${dbTurnId}`,
      );
    } catch {
      // If the API call fails, still remove from local state for responsiveness
    }

    setMessagesByConv((prev) => ({
      ...prev,
      [activeConversationId]: (prev[activeConversationId] || []).filter(
        (m) => m.id !== turnToDelete.msgId,
      ),
    }));
    delete turnDbIdByMsgId.current[turnToDelete.msgId];
    queryClient.invalidateQueries({ queryKey: ["conversation", activeConversationId] });
    setTurnToDelete(null);
  }, [turnToDelete, activeConversationId, queryClient]);

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const first = message.content?.[0];
      const input =
        first && typeof first !== "string" && first.type === "text" ? first.text : "";
      if (!input) return;

      // Capture the conversation ID at send time — updates target this conversation
      // even if the user navigates away mid-stream
      const convId = activeConversationId;

      const updateConvMessages = (updater: (prev: ThreadMessageLike[]) => ThreadMessageLike[]) => {
        setMessagesByConv((prev) => ({
          ...prev,
          [convId!]: updater(prev[convId!] || []),
        }));
      };

      setError(null);
      setLastArtifacts([]);
      setArtifactMessageId(null);

      const userMsg: ThreadMessageLike = {
        id: generateId(),
        role: "user",
        content: [{ type: "text", text: input }],
      };

      const assistantId = generateId();
      const assistantMsg: ThreadMessageLike = {
        id: assistantId,
        role: "assistant",
        content: [{ type: "text", text: "" }],
      };

      updateConvMessages((prev) => [...prev, userMsg, assistantMsg]);
      setRunningConvId(convId);

      const controller = new AbortController();
      abortRef.current = controller;

      let fullText = "";

      await streamChat({
        message: input,
        conversationId: convId ?? undefined,
        agent: selectedAgent || undefined,
        systemMessage: systemMessage || undefined,
        filterTools: selectedSkills.length > 0 ? selectedSkills : undefined,
        abortSignal: controller.signal,
        onSessionReady: (sessionId) => {
          if (!convId) {
            loadedConversationIds.current.add(sessionId);
            navigate(`/admin/chat/${encodeURIComponent(sessionId)}`, { replace: true });
          }
        },
        onEvent: (event) => {
          let chunk = "";
          if (event.type === "text" && event.content) {
            chunk = event.content;
          } else if (event.type === "thinking" && event.content) {
            chunk = `\n> ${event.content}\n`;
          } else if (event.type === "tool_use" && event.content) {
            const label =
              event.content.length > 60
                ? event.content.slice(0, 60) + "..."
                : event.content;
            chunk = `\n\`\`\`\n${label}\n\`\`\`\n`;
          } else if (event.type === "tool_result" && event.content) {
            const truncated =
              event.content.length > 600
                ? event.content.slice(0, 600) + "..."
                : event.content;
            chunk = `\n\`\`\`\n${truncated}\n\`\`\`\n`;
          } else if (event.type === "status" && event.content) {
            chunk = `\n_${event.content}_\n`;
          }

          if (chunk) {
            fullText += chunk;
            updateConvMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: [{ type: "text" as const, text: fullText }] }
                  : m,
              ),
            );
          }
        },
        onError: (err) => {
          if (convId === activeConversationId) setError(err.message);
          setRunningConvId((prev) => (prev === convId ? null : prev));
          abortRef.current = null;
        },
        onDone: (_fullText, artifacts, sessionId) => {
          setRunningConvId((prev) => (prev === convId ? null : prev));
          abortRef.current = null;
          if (artifacts.length > 0 && convId === activeConversationId) {
            setLastArtifacts(artifacts);
            setArtifactMessageId(assistantId);
          }
          if (sessionId && !convId) {
            loadedConversationIds.current.add(sessionId);
            navigate(`/admin/chat/${encodeURIComponent(sessionId)}`, { replace: true });
          }
          queryClient.invalidateQueries({ queryKey: ["conversations"] });
        },
      });
    },
    [activeConversationId, navigate, queryClient, selectedAgent, systemMessage, selectedSkills],
  );

  const onNewRef = useRef(onNew);
  onNewRef.current = onNew;

  const handleCancel = useCallback(async () => {
    abortRef.current?.abort();
    abortRef.current = null;
    if (runningConvId) {
      setMessagesByConv((prev) => {
        const msgs = prev[runningConvId];
        if (!msgs) return prev;
        const last = msgs[msgs.length - 1];
        if (
          last?.role === "assistant" &&
          last.content &&
          typeof last.content !== "string" &&
          last.content[0]?.type === "text" &&
          !last.content[0].text
        ) {
          return { ...prev, [runningConvId]: msgs.slice(0, -1) };
        }
        return prev;
      });
    }
    setRunningConvId(null);
  }, [runningConvId]);

  const runtime = useExternalStoreRuntime({
    isRunning,
    isSendDisabled: isRunning,
    messages,
    convertMessage: (msg: ThreadMessageLike) => msg,
    onNew,
    onCancel: handleCancel,
  });

  const showEmptyState = !activeConversationId && messages.length === 0;

  return (
    <div className="flex-1 -m-6 flex overflow-hidden">
      <ConversationSidebar
        activeId={activeConversationId}
        runningId={runningConvId}
        onNew={handleNewChat}
        onDelete={handleDelete}
      />

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden bg-background">
        {error && (
          <div className="px-4 py-2 text-xs text-red-700 bg-red-50 border-b border-red-200 shrink-0">
            {error}
          </div>
        )}

        <AssistantRuntimeProvider runtime={runtime}>
          <ThreadPrimitive.Root className="grid grid-rows-[1fr_auto] flex-1 min-h-0">
            <ThreadPrimitive.Viewport
              className="overflow-y-auto min-h-0"
              autoScroll
            >
              <div className="max-w-4xl mx-auto w-full px-4 lg:px-6 py-4 lg:py-6">
                {showEmptyState && (
                  <div className="flex flex-col items-center justify-center text-center min-h-[400px]">
                    <h1 className="text-2xl font-semibold text-foreground mb-2">
                      {t("chat.emptyStateTitle")}
                    </h1>
                    <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
                      {t("chat.emptyStateDesc")}
                    </p>
                  </div>
                )}

                <div className="space-y-4">
                  <ThreadPrimitive.Messages>
                    {({ message }) => (
                      <div
                        className={`message-enter flex ${
                          message.role === "user" ? "justify-end" : "justify-start"
                        }`}
                      >
                        <div
                          className={`message-bubble group relative ${
                            message.role === "user"
                              ? "message-bubble-user"
                              : "message-bubble-assistant"
                          }`}
                        >
                          <MessagePrimitive.Content
                            components={{
                              Text: ({ text, status }) => {
                                if (status?.type === "running" && !text) {
                                  return (
                                    <div className="loading-dots flex gap-1 py-1">
                                      <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                                      <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                                      <span className="w-2 h-2 bg-foreground/40 rounded-full" />
                                    </div>
                                  );
                                }
                                // Render as plain text during streaming to avoid
                                // markdown re-parse jank on every incremental chunk.
                                if (status?.type === "running") {
                                  return <span className="whitespace-pre-wrap">{text}</span>;
                                }
                                return <MarkdownTextPrimitive {...({ children: text } as any)} />;
                              },
                            }}
                          />
                          {message.id === artifactMessageId && lastArtifacts.length > 0 && (
                            <div className="mt-2 pt-2 border-t space-y-2">
                              {lastArtifacts.map((a, i) => (
                                <ChatArtifactPreview key={i} artifact={a} />
                              ))}
                            </div>
                          )}
                          <button
                            type="button"
                            className="absolute top-1 right-1 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-muted transition-opacity"
                            onClick={(e) => {
                              e.stopPropagation();
                              setTurnToDelete({ msgId: message.id as string });
                            }}
                            title={t("chat.deleteTurn")}
                          >
                            <Trash2 className="w-3 h-3 text-muted-foreground hover:text-destructive transition-colors" />
                          </button>
                        </div>
                      </div>
                    )}
                  </ThreadPrimitive.Messages>

                  <div ref={scrollAnchorRef} />
                </div>
              </div>
            </ThreadPrimitive.Viewport>

            {/* Composer — multi-line input, config menu, templates */}
            <div className="border-t border-border bg-background shrink-0">
              <div className="max-w-4xl mx-auto w-full px-4 lg:px-6 py-3 lg:py-4 space-y-3">
                {/* Selected config tags */}
                {(selectedAgent || systemMessage || selectedSkills.length > 0) && (
                  <div className="flex flex-wrap gap-1.5">
                    {selectedAgent && (
                      <span className="inline-flex items-center gap-1 rounded-md bg-primary/10 text-primary text-[11px] px-2 py-0.5">
                        {t("agents.title")}: {selectedAgent}
                        <button onClick={() => handleSetAgent("")}><X className="w-3 h-3" /></button>
                      </span>
                    )}
                    {systemMessage && (
                      <span className="inline-flex items-center gap-1 rounded-md bg-primary/10 text-primary text-[11px] px-2 py-0.5">
                        {t("agents.systemPrompt")}
                        <button onClick={() => handleSetSystemMessage("")}><X className="w-3 h-3" /></button>
                      </span>
                    )}
                    {selectedSkills.map((s) => (
                      <span key={s} className="inline-flex items-center gap-1 rounded-md bg-primary/10 text-primary text-[11px] px-2 py-0.5">
                        {s}
                        <button onClick={() => handleSetSkills(selectedSkills.filter((x) => x !== s))}><X className="w-3 h-3" /></button>
                      </span>
                    ))}
                  </div>
                )}

                {/* Input area with + config button */}
                <div className="flex gap-3 items-start">
                  {/* + Config button */}
                  <button
                    type="button"
                    onClick={() => setShowConfigMenu(!showConfigMenu)}
                    className="inline-flex items-center justify-center rounded-lg p-2 text-sm font-medium border border-input hover:bg-muted transition-colors shrink-0 mt-0.5"
                  >
                    <Plus className="w-4 h-4" />
                  </button>

                  {/* Textarea */}
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={t("chat.typeMessage")}
                    disabled={isRunning}
                    rows={1}
                    className="flex-1 min-w-0 rounded-lg border border-input bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground disabled:opacity-50 resize-none field-sizing-content"
                    style={{ minHeight: "44px", maxHeight: "200px" }}
                    onInput={(e) => {
                      const el = e.currentTarget;
                      el.style.height = "auto";
                      el.style.height = Math.min(el.scrollHeight, 200) + "px";
                    }}
                  />

                  {/* Send / Cancel */}
                  {isRunning ? (
                    <button
                      type="button"
                      className="inline-flex items-center justify-center rounded-lg p-2.5 text-sm font-medium border border-input hover:bg-muted transition-colors shrink-0 mt-0.5"
                      onClick={handleCancel}
                    >
                      <Square className="w-4 h-4" />
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSend}
                      disabled={!inputText.trim()}
                      className="inline-flex items-center justify-center rounded-lg p-2.5 text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors shrink-0 disabled:opacity-50 mt-0.5"
                    >
                      <ArrowUp className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Config panel — shown when + is clicked */}
                {showConfigMenu && (
                  <ComposerConfigPanel
                    selectedAgent={selectedAgent}
                    onAgentChange={handleSetAgent}
                    systemMessage={systemMessage}
                    onSystemMessageChange={handleSetSystemMessage}
                    selectedSkills={selectedSkills}
                    onSkillsChange={handleSetSkills}
                    onClose={() => setShowConfigMenu(false)}
                  />
                )}

                {/* Template suggestions */}
                {!isRunning && templates && templates.length > 0 && (
                  <div className="flex gap-1.5 overflow-x-auto pb-1">
                    {templates.map((tpl) => (
                      <button
                        key={tpl.id}
                        type="button"
                        onClick={() => handleApplyTemplate(tpl)}
                        className="shrink-0 rounded-full border border-border px-3 py-1 text-[12px] text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                      >
                        {t(`templates.${tpl.id}`) === `templates.${tpl.id}` ? tpl.label || tpl.id : t(`templates.${tpl.id}`)}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </ThreadPrimitive.Root>
        </AssistantRuntimeProvider>

        <ConfirmDialog
          open={turnToDelete !== null}
          title={t("chat.deleteTurn")}
          description={t("chat.deleteTurnConfirm")}
          destructive
          onConfirm={handleDeleteTurn}
          onCancel={() => setTurnToDelete(null)}
        />
      </main>
    </div>
  );
}

function ComposerConfigPanel({
  selectedAgent,
  onAgentChange,
  systemMessage,
  onSystemMessageChange,
  selectedSkills,
  onSkillsChange,
  onClose,
}: {
  selectedAgent: string;
  onAgentChange: (v: string) => void;
  systemMessage: string;
  onSystemMessageChange: (v: string) => void;
  selectedSkills: string[];
  onSkillsChange: (v: string[]) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const { data: agents } = useAgents();
  const { data: skills } = useSkills();

  // Local textarea state + debounced sync to parent
  const [localSysMsg, setLocalSysMsg] = useState(systemMessage);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync from parent prop when it changes externally (e.g. template apply)
  useEffect(() => {
    setLocalSysMsg(systemMessage);
  }, [systemMessage]);

  const handleSysMsgChange = useCallback((v: string) => {
    setLocalSysMsg(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => onSystemMessageChange(v), 400);
  }, [onSystemMessageChange]);

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  const agentList: string[] = agents
    ? agents.map((a) => a.name).filter(Boolean)
    : [];
  const skillList: string[] = skills
    ? (skills as Array<Record<string, unknown>>).map((s) => s.name as string).filter(Boolean)
    : [];

  return (
    <div className="rounded-lg border bg-card p-4 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">{t("chat.configure")}</span>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Agent */}
      <div>
        <label className="text-[11px] font-medium text-muted-foreground">{t("agents.title")}</label>
        <select
          value={selectedAgent}
          onChange={(e) => onAgentChange(e.target.value)}
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
        >
          <option value="">Default</option>
          {agentList.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
      </div>

      {/* System prompt */}
      <div>
        <label className="text-[11px] font-medium text-muted-foreground">{t("agents.systemPrompt")}</label>
        <textarea
          value={localSysMsg}
          onChange={(e) => handleSysMsgChange(e.target.value)}
          rows={3}
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm resize-y focus:outline-none focus:ring-2 focus:ring-ring"
          placeholder="Instructions for the AI..."
        />
      </div>

      {/* Skills */}
      <div>
        <label className="text-[11px] font-medium text-muted-foreground">{t("sidebar.skills")}</label>
        {skillList.length === 0 ? (
          <p className="text-xs text-muted-foreground mt-1">No skills available</p>
        ) : (
          <div className="mt-1 flex flex-wrap gap-1.5">
            {skillList.map((s) => {
              const active = selectedSkills.includes(s);
              return (
                <button
                  key={s}
                  type="button"
                  onClick={() => onSkillsChange(active ? selectedSkills.filter((x) => x !== s) : [...selectedSkills, s])}
                  className={`rounded-md px-2 py-0.5 text-[11px] transition-colors ${
                    active ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:bg-secondary"
                  }`}
                >
                  {s}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function ChatArtifactPreview({ artifact }: { artifact: ChatArtifact }) {
  const mediaType = artifact.media_type;
  const name = artifact.path.split("/").pop() || artifact.path;

  if (mediaType.startsWith("image/")) {
    return (
      <div className="rounded-md overflow-hidden border">
        <img
          src={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`}
          alt={name}
          className="max-w-full h-auto max-h-64 object-contain"
        />
        <div className="px-2 py-1 text-[10px] text-muted-foreground bg-muted/50">
          {name} {artifact.size ? `(${(artifact.size / 1024).toFixed(1)} KB)` : ""}
        </div>
      </div>
    );
  }

  if (mediaType.startsWith("audio/")) {
    return (
      <div className="rounded-md border p-2">
        <audio controls className="w-full h-8">
          <source src={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`} type={mediaType} />
        </audio>
        <div className="text-[10px] text-muted-foreground mt-1">{name}</div>
      </div>
    );
  }

  if (mediaType.startsWith("video/")) {
    return (
      <div className="rounded-md overflow-hidden border">
        <video controls className="max-w-full max-h-48">
          <source src={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`} type={mediaType} />
        </video>
        <div className="px-2 py-1 text-[10px] text-muted-foreground bg-muted/50">{name}</div>
      </div>
    );
  }

  return (
    <a
      href={`/admin/api/files?path=${encodeURIComponent(artifact.path)}`}
      className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm hover:bg-muted/50 transition-colors"
      download={name}
    >
      <span className="text-muted-foreground">📄</span>
      <span className="flex-1 truncate">{name}</span>
      {artifact.size && (
        <span className="text-[10px] text-muted-foreground">
          {(artifact.size / 1024).toFixed(1)} KB
        </span>
      )}
    </a>
  );
}
