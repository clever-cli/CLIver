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
import { ArrowUp, Square, Play, FileText, Brain, Package, ChevronDown, ChevronRight, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useLab, useLabGoldenTests, useRunGoldenTests, type TestRunResult } from "@/hooks/use-api";
import { useConversation } from "@/hooks/use-conversations";
import { streamChat } from "@/lib/chat-stream";
import { LabHeader } from "@/components/lab/LabHeader";
import { LabConfigPanel } from "@/components/lab/LabConfigPanel";
import { GoldenTestCard } from "@/components/lab/GoldenTestCard";
import { useTranslation } from "@/i18n";

function generateId(): string {
  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

function TraceJsonBlock({ label, data }: { label: string; data: unknown }) {
  const [open, setOpen] = useState(false);
  const jsonStr = typeof data === "string" ? data : JSON.stringify(data, null, 2);
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {label}
      </button>
      {open && (
        <pre className="mt-1 p-2 bg-muted/50 rounded text-[11px] max-h-60 overflow-auto whitespace-pre-wrap">
          {jsonStr}
        </pre>
      )}
    </div>
  );
}

function TraceIteration({ iteration, index }: { iteration: Record<string, unknown>; index: number }) {
  const [open, setOpen] = useState(false);
  const model = iteration.model as string;
  const provider = iteration.provider as string;
  return (
    <div className="border rounded-md text-xs">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full px-2 py-1.5 hover:bg-accent/50 transition-colors rounded-md text-left"
      >
        {open ? <ChevronDown className="w-3 h-3 shrink-0" /> : <ChevronRight className="w-3 h-3 shrink-0" />}
        <Badge variant="secondary" className="text-[10px]">Iteration {index + 1}</Badge>
        <span className="text-muted-foreground">{model}</span>
        {provider && <span className="text-muted-foreground">({provider})</span>}
      </button>
      {open && (
        <div className="px-2 pb-2 space-y-1.5">
          {iteration.request != null && <TraceJsonBlock label="Request" data={iteration.request} />}
          {iteration.response != null && <TraceJsonBlock label="Raw Response" data={iteration.response} />}
          {iteration.message != null && <TraceJsonBlock label="Parsed Message" data={iteration.message} />}
        </div>
      )}
    </div>
  );
}

export default function LabChatPage() {
  const { t } = useTranslation();
  const { labId, sessionId: paramSessionId } = useParams<{ labId: string; sessionId?: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: labDetail, isLoading: labLoading } = useLab(labId);
  const { data: tests } = useLabGoldenTests(labId);
  const runTests = useRunGoldenTests(labId!);

  // The lab's session is created at lab creation time (1:1). Use the session_id
  // from the URL param, or fall back to the lab's default session_id.
  const activeSessionId = paramSessionId || labDetail?.session_id || null;

  // Redirect to session URL once we know the session_id
  useEffect(() => {
    if (labDetail?.session_id && !paramSessionId) {
      navigate(`/admin/labs/${labId}/chat/${labDetail.session_id}`, { replace: true });
    }
  }, [labDetail?.session_id, paramSessionId, labId, navigate]);

  const { data: conversationDetail } = useConversation(activeSessionId);

  const [messagesByConv, setMessagesByConv] = useState<Record<string, ThreadMessageLike[]>>({});
  const [traceByMsgId, setTraceByMsgId] = useState<Record<string, Array<Record<string, unknown>>>>({});
  const [inspectTrace, setInspectTrace] = useState<Array<Record<string, unknown>> | null>(null);
  const [runningConvId, setRunningConvId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const loadedSessionIds = useRef<Set<string>>(new Set());
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  const messages = activeSessionId ? (messagesByConv[activeSessionId] || []) : [];
  const isRunning = !!(runningConvId && runningConvId === activeSessionId);

  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [selectedMCPServerIds, setSelectedMCPServerIds] = useState<string[]>([]);
  const [savingConfig, setSavingConfig] = useState(false);
  const [testResults, setTestResults] = useState<TestRunResult[] | null>(null);

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

  // Invalidate cached conversation data on mount so we always load fresh options.
  useEffect(() => {
    if (activeSessionId) {
      queryClient.invalidateQueries({ queryKey: ["conversation", activeSessionId] });
    }
  }, [activeSessionId, queryClient]);

  // Load config from session options — follows the same pattern as the existing
  // chat.tsx page, using React Query's conversationDetail for reliable loading.
  const lastLoadedSessionId = useRef<string | null>(null);
  useEffect(() => {
    if (!activeSessionId) {
      setSelectedModel("");
      setSystemPrompt("");
      setSelectedSkills([]);
      setSelectedMCPServerIds([]);
      lastLoadedSessionId.current = null;
      return;
    }
    const dataId = conversationDetail?.session?.id;
    if (!dataId || dataId !== activeSessionId) return;
    if (lastLoadedSessionId.current === dataId) return;
    lastLoadedSessionId.current = dataId;

    const opts = (conversationDetail?.session?.options as Record<string, unknown>) || {};
    if (opts.model) setSelectedModel(String(opts.model));
    if (opts.system_prompt) setSystemPrompt(String(opts.system_prompt));
    if (opts.skills) setSelectedSkills(Array.isArray(opts.skills) ? (opts.skills as string[]) : []);
    if (opts.mcp_servers) setSelectedMCPServerIds(Array.isArray(opts.mcp_servers) ? (opts.mcp_servers as string[]) : []);
  }, [activeSessionId, conversationDetail]);

  // Clear load tracker when leaving
  useEffect(() => {
    return () => { lastLoadedSessionId.current = null; };
  }, [activeSessionId]);

  // Load turns into messages when conversationDetail changes
  useEffect(() => {
    if (activeSessionId && runningConvId === activeSessionId) return;
    if (activeSessionId && conversationDetail?.turns) {
      const dataId = conversationDetail.session?.id;
      if (dataId && dataId !== activeSessionId) return;
      if (loadedSessionIds.current.has(activeSessionId)) return;
      loadedSessionIds.current.add(activeSessionId);
      setMessagesByConv((prev) => ({
        ...prev,
        [activeSessionId]: conversationDetail.turns.map((turn) => ({
          id: generateId(),
          role: turn.role as "user" | "assistant",
          content: [{ type: "text" as const, text: turn.content }],
        })),
      }));
    }
  }, [activeSessionId, conversationDetail, runningConvId]);

  useEffect(() => {
    if (scrollAnchorRef.current) {
      scrollAnchorRef.current.scrollIntoView({ behavior: "instant" as ScrollBehavior });
    }
  }, [messages]);

  const BUILTIN_SKILLS = ["brainstorm", "write-plan", "execute-plan"];

  const handleSaveConfig = useCallback(async () => {
    if (!activeSessionId) return;
    setSavingConfig(true);
    const skills = [...new Set([...BUILTIN_SKILLS, ...selectedSkills])];
    try {
      await fetch(
        `/admin/api/labs/${encodeURIComponent(labId!)}/chat/${encodeURIComponent(activeSessionId)}`,
        {
          method: "PATCH",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            options: {
              model: selectedModel || null,
              system_prompt: systemPrompt || null,
              skills,
              mcp_servers: selectedMCPServerIds,
            },
          }),
        },
      );
      queryClient.invalidateQueries({ queryKey: ["conversation", activeSessionId] });
    } catch {}
    setSavingConfig(false);
  }, [activeSessionId, labId, selectedModel, systemPrompt, selectedSkills, selectedMCPServerIds, queryClient]);

  const handleSend = useCallback(() => {
    const text = inputText.trim();
    if (!text || isRunning) return;
    onNewRef.current?.({
      role: "user",
      content: [{ type: "text" as const, text }],
      parentId: null,
      sourceId: null,
      runConfig: undefined,
      metadata: undefined,
      createdAt: new Date(),
    } as unknown as AppendMessage);
    setInputText("");
  }, [inputText, isRunning]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
      if (e.nativeEvent.isComposing) return;
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleCancel = useCallback(async () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunningConvId(null);
  }, []);

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const first = message.content?.[0];
      const input = first && typeof first !== "string" && first.type === "text" ? first.text : "";
      if (!input) return;

      const convId = activeSessionId;
      const convLabId = labId!;

      const updateConvMessages = (updater: (prev: ThreadMessageLike[]) => ThreadMessageLike[]) => {
        setMessagesByConv((prev) => ({
          ...prev,
          [convId!]: updater(prev[convId!] || []),
        }));
      };

      setError(null);
      setTestResults(null);

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
        apiPath: `/admin/api/labs/${encodeURIComponent(convLabId)}/chat/${encodeURIComponent(convId!)}`,
        model: selectedModel || undefined,
        systemMessage: systemPrompt || undefined,
        filterTools: undefined,
        conversationId: convId ?? undefined,
        abortSignal: controller.signal,
        extraBody: { mcp_server_ids: selectedMCPServerIds },
        onSessionReady: (_sessionId) => {
          queryClient.invalidateQueries({ queryKey: ["conversations"] });
        },
        onEvent: (event) => {
          let chunk = "";
          if (event.type === "text" && event.content) {
            chunk = event.content;
          } else if (event.type === "thinking" && event.content) {
            chunk = `\n> ${event.content}\n`;
          } else if (event.type === "tool_use" && event.content) {
            chunk = `\n\`\`\`\n${event.content.length > 60 ? event.content.slice(0, 60) + "..." : event.content}\n\`\`\`\n`;
          } else if (event.type === "tool_result" && event.content) {
            const truncated = event.content.length > 300 ? event.content.slice(0, 300) + "..." : event.content;
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
          setError(err.message);
          setRunningConvId(null);
        },
        onDone: (_fullText, _artifacts, _sessionId, llmTrace) => {
          if (llmTrace) {
            setTraceByMsgId((prev) => ({ ...prev, [assistantId]: llmTrace }));
          }
          setRunningConvId(null);
          queryClient.invalidateQueries({ queryKey: ["conversations"] });
          queryClient.invalidateQueries({ queryKey: ["conversation", convId] });
        },
      });
    },
    [activeSessionId, labId, navigate, queryClient, selectedModel, systemPrompt, selectedSkills, selectedMCPServerIds],
  );

  const onNewRef = useRef(onNew);
  onNewRef.current = onNew;

  const runtime = useExternalStoreRuntime({
    isRunning,
    isSendDisabled: isRunning,
    messages,
    convertMessage: (msg: ThreadMessageLike) => msg,
    onNew,
    onCancel: handleCancel,
  });

  if (labLoading) {
    return <p className="text-sm text-muted-foreground p-4">{t("common.loading")}</p>;
  }

  if (!labDetail) {
    return <p className="text-sm text-muted-foreground p-4">{t("labs.notFound")}</p>;
  }

  const lab = labDetail.lab;
  const showEmptyState = !activeSessionId && messages.length === 0;

  const traceMsgIds = new Set(Object.keys(traceByMsgId));
  // Also include trace from conversationDetail turns loaded from DB
  const dbTurns = (conversationDetail?.turns || []) as unknown as Array<Record<string, unknown>>;
  const dbTraceByIndex: Record<number, Array<Record<string, unknown>>> = {};
  dbTurns.forEach((turn, i) => {
    const msg = turn.message as Record<string, unknown> | undefined;
    const vendorExt = (msg?.vendor_ext || {}) as Record<string, unknown>;
    const trace = vendorExt.__llm_trace__ as Array<Record<string, unknown>> | undefined;
    if (trace) dbTraceByIndex[i] = trace;
  });

  return (
    <div className="flex flex-col h-full -m-6">
      <LabHeader
        title={lab.title}
        description={lab.description}
        breadcrumb={
          <>
            <button onClick={() => navigate("/admin/labs")} className="hover:text-foreground transition-colors">
              {t("labs.title")}
            </button>
            <span className="mx-0.5">›</span>
            <button onClick={() => navigate(`/admin/labs/${labId}`)} className="hover:text-foreground transition-colors">
              {lab.title}
            </button>
          </>
        }
      />

      <div className="flex flex-1 min-h-0">
        {/* LEFT PANEL */}
        <div className="w-[240px] min-w-[240px] border-r bg-card overflow-hidden">
          <LabConfigPanel
            selectedModel={selectedModel}
            systemPrompt={systemPrompt}
            selectedSkills={selectedSkills}
            selectedMCPServerIds={selectedMCPServerIds}
            onModelChange={setSelectedModel}
            onSystemPromptChange={setSystemPrompt}
            onSkillsChange={setSelectedSkills}
            onMCPServersChange={setSelectedMCPServerIds}
            onSave={handleSaveConfig}
            saving={savingConfig}
          />
        </div>

        {/* MAIN PANEL */}
        <div className="flex-1 flex flex-col min-w-0 bg-background">
          {error && (
            <div className="px-4 py-2 text-xs text-red-700 bg-red-50 border-b border-red-200 shrink-0">
              {error}
            </div>
          )}
          <AssistantRuntimeProvider runtime={runtime}>
            <ThreadPrimitive.Root className="grid grid-rows-[1fr_auto] flex-1 min-h-0">
              <ThreadPrimitive.Viewport className="overflow-y-auto min-h-0" autoScroll>
                <div className="max-w-4xl mx-auto w-full px-4 lg:px-6 py-4 lg:py-6">
                  {showEmptyState && (
                    <div className="flex flex-col items-center justify-center text-center min-h-[400px]">
                      <h1 className="text-2xl font-semibold text-foreground mb-2">
                        {lab.title}
                      </h1>
                      <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
                        {lab.description || t("lab.chat")}
                      </p>
                    </div>
                  )}
                  <div className="space-y-4">
                    <ThreadPrimitive.Messages>
                      {({ message }: { message: { id?: string; role: string } }) => {
                        const liveTrace = traceByMsgId[message.id as string];
                        // Match DB trace by message index in the visible list
                        const msgIndex = messages.findIndex((m: { id?: string }) => m.id === message.id);
                        const dbTrace = msgIndex >= 0 ? dbTraceByIndex[msgIndex] : undefined;
                        const trace = liveTrace || dbTrace;
                        const hasTrace = trace && trace.length > 0;
                        return (
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
                            } ${hasTrace ? "cursor-pointer" : ""}`}
                            onClick={hasTrace ? () => setInspectTrace(trace) : undefined}
                            title={hasTrace ? "Click to inspect provider communication" : undefined}
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
                          {hasTrace && (
                            <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                              <Badge variant="outline" className="text-[9px] px-1 py-0">trace</Badge>
                            </div>
                          )}
                          </div>
                        </div>
                        );
                      }}
                    </ThreadPrimitive.Messages>
                    <div ref={scrollAnchorRef} />
                  </div>

                  {/* Trace inspection dialog */}
                  {inspectTrace && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setInspectTrace(null)}>
                      <div className="bg-card rounded-lg shadow-xl max-w-3xl w-full max-h-[80vh] overflow-y-auto m-4" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-between px-4 py-3 border-b sticky top-0 bg-card z-10">
                          <h3 className="text-sm font-semibold">Provider Communication ({inspectTrace.length} iteration{inspectTrace.length > 1 ? "s" : ""})</h3>
                          <button onClick={() => setInspectTrace(null)} className="p-1 hover:bg-muted rounded">
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                        <div className="p-4 space-y-2">
                          {inspectTrace.map((iteration, i) => (
                            <TraceIteration key={i} iteration={iteration} index={i} />
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </ThreadPrimitive.Viewport>

              {/* Input box */}
              <div className="border-t bg-card shrink-0 px-4 py-3">
                <div className="max-w-3xl mx-auto w-full flex gap-2 items-end">
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={t("lab.typeMessage")}
                    disabled={isRunning}
                    rows={3}
                    className="flex-1 min-w-0 rounded-lg border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring placeholder:text-muted-foreground disabled:opacity-50 resize-none"
                    style={{ minHeight: "72px", maxHeight: "200px" }}
                    onInput={(e) => {
                      const el = e.currentTarget;
                      el.style.height = "auto";
                      el.style.height = Math.min(el.scrollHeight, 160) + "px";
                    }}
                  />
                  {isRunning ? (
                    <button
                      type="button"
                      onClick={handleCancel}
                      className="shrink-0 rounded-lg p-2 border border-input hover:bg-muted transition-colors"
                    >
                      <Square className="w-4 h-4" />
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSend}
                      disabled={!inputText.trim()}
                      className="shrink-0 rounded-lg p-2 bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
                    >
                      <ArrowUp className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            </ThreadPrimitive.Root>
          </AssistantRuntimeProvider>
        </div>

        {/* RIGHT PANEL */}
        <div className="w-[180px] min-w-[180px] border-l bg-card overflow-y-auto p-3 space-y-3">
          <div>
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
              {t("lab.actions")}
            </p>
            <button
              type="button"
              onClick={async () => {
                const result = await runTests.mutateAsync();
                setTestResults(result.results);
              }}
              disabled={runTests.isPending || !tests || tests.length === 0}
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5" />
              {t("lab.runTests")}
            </button>
            <button
              type="button"
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 text-muted-foreground"
            >
              <FileText className="w-3.5 h-3.5" />
              {t("lab.viewReport")}
            </button>
            <button
              type="button"
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 text-muted-foreground"
              title={t("lab.comingSoon")}
            >
              <Brain className="w-3.5 h-3.5" />
              {t("lab.distillSkill")}
            </button>
            <button
              type="button"
              className="w-full text-left rounded-md px-2 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 text-muted-foreground"
              title={t("lab.comingSoon")}
            >
              <Package className="w-3.5 h-3.5" />
              {t("lab.generatePackage")}
            </button>
          </div>

          {tests && tests.length > 0 && (
            <div>
              <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2">
                {t("lab.benchmarks")} ({tests.length})
              </p>
              <div className="space-y-1.5">
                {tests.map((test) => (
                  <GoldenTestCard
                    key={test.id}
                    name={test.name}
                    input={test.input}
                    expectedOutput={test.expected_output}
                    result={testResults?.find((r) => r.test_id === test.id)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
