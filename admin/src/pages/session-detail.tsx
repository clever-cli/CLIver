import { useState } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useSessionTurns, useDeleteSession } from "@/hooks/use-api";
import { apiDelete } from "@/lib/api";
import { ConfirmDialog } from "@/components/confirm-dialog";
import {
  ArrowLeft, Trash2, User, Bot, Wrench, Brain, ChevronDown, ChevronRight, Image as ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MarkdownView } from "@/components/markdown-view";
import { useTranslation } from "@/i18n";

interface TurnMessage {
  role: string;
  content: string | null;
  id?: string;
  tool_calls?: Array<{ id: string; name: string; args: Record<string, unknown> }>;
  vendor_ext?: Record<string, unknown>;
}

interface TurnData {
  id?: number;
  role: string;
  content: string;
  timestamp?: string;
  type?: string;
  additional_kwargs?: Record<string, unknown>;
  tool_calls?: Array<{ name: string; args: Record<string, unknown>; id?: string }>;
  tool_call_id?: string;
  tool_name?: string;
  media?: Array<{ type: string; path: string; mime?: string }>;
  message?: TurnMessage;
}

function ReasoningBlock({ content }: { content: string }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  if (!content) return null;
  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <Brain className="w-3 h-3" />
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {t("sessions.thinking")}
      </button>
      {open && (
        <div className="mt-1 pl-4 border-l-2 border-muted text-xs text-muted-foreground italic whitespace-pre-wrap max-h-60 overflow-auto">
          {content}
        </div>
      )}
    </div>
  );
}

function ToolCallsBlock({ calls }: { calls: TurnData["tool_calls"] }) {
  if (!calls || calls.length === 0) return null;
  return (
    <div className="mt-2 space-y-1">
      {calls.map((tc, i) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          <Wrench className="w-3 h-3 text-muted-foreground" />
          <Badge variant="outline" className="text-[10px]">{tc.name}</Badge>
          {tc.args && Object.keys(tc.args).length > 0 && (
            <span className="text-muted-foreground font-mono truncate max-w-[400px]">
              {JSON.stringify(tc.args).slice(0, 100)}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

function MediaBlock({ media, sessionSource, sessionId }: {
  media: TurnData["media"]; sessionSource?: string; sessionId?: string;
}) {
  if (!media || media.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {media.map((m, i) => {
        const isImage = m.type === "image" || m.mime?.startsWith("image/");
        if (isImage && m.path) {
          const src = m.path.startsWith("http") ? m.path : `/admin/api/sessions/${sessionSource}/${sessionId}/media/${m.path}`;
          return <img key={i} src={src} alt="" className="max-w-xs max-h-48 rounded-md border" />;
        }
        return (
          <div key={i} className="flex items-center gap-1 text-xs text-muted-foreground">
            <ImageIcon className="w-3 h-3" />
            <span>{m.path || m.type}</span>
          </div>
        );
      })}
    </div>
  );
}

function ProviderDataPanel({ vendorExt }: { vendorExt?: Record<string, unknown> }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const trace = vendorExt?.__llm_trace__ as Array<Record<string, unknown>> | undefined;
  if (!trace || trace.length === 0) return null;

  return (
    <div className="mt-3 border-t pt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        {t("trace.providerCommunication", { count: trace.length, plural: trace.length > 1 ? "s" : "" })}
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          {trace.map((iteration, i) => (
            <IterationDetail key={i} iteration={iteration} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}

function IterationDetail({ iteration, index }: { iteration: Record<string, unknown>; index: number }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const model = iteration.model as string;
  const provider = iteration.provider as string;
  const request = iteration.request;
  const response = iteration.response;
  const message = iteration.message;

  return (
    <div className="border rounded-md text-xs">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full px-2 py-1.5 hover:bg-accent/50 transition-colors rounded-md text-left"
      >
        {open ? <ChevronDown className="w-3 h-3 shrink-0" /> : <ChevronRight className="w-3 h-3 shrink-0" />}
        <Badge variant="secondary" className="text-[10px]">{t("trace.iteration", { n: index + 1 })}</Badge>
        <span className="text-muted-foreground">{model}</span>
        {provider && <span className="text-muted-foreground">({provider})</span>}
      </button>
      {open && (
        <div className="px-2 pb-2 space-y-1.5">
          {request != null && (
            <JsonBlock label={t("trace.request")} data={request} />
          )}
          {response != null && (
            <JsonBlock label={t("trace.rawResponse")} data={response} />
          )}
          {message != null && (
            <JsonBlock label={t("trace.parsedMessage")} data={message} />
          )}
        </div>
      )}
    </div>
  );
}

function JsonBlock({ label, data }: { label: string; data: unknown }) {
  const [open, setOpen] = useState(false);
  const jsonStr = typeof data === "string" ? data : JSON.stringify(data, null, 2);

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
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

function TurnCard({ turn, sessionId, onDelete }: { turn: TurnData; sessionId?: string; onDelete?: () => void }) {
  const { t } = useTranslation();
  const isUser = turn.role === "user";
  const isTool = turn.role === "tool";
  const msg = turn.message;
  const vendorExt = msg?.vendor_ext;
  const reasoning = (vendorExt?.reasoning_content as string) || (turn.additional_kwargs?.reasoning_content as string);
  const messageToolCalls = msg?.tool_calls?.map((tc) => ({ name: tc.name, args: tc.args, id: tc.id }));
  const displayToolCalls = messageToolCalls || turn.tool_calls;

  if (isTool) {
    return (
      <Card className="border-l-4 border-l-amber-500/50 group relative">
        <CardContent className="pt-3 pb-3">
          <div className="flex items-start gap-3">
            <div className="flex items-center justify-center w-7 h-7 rounded-full shrink-0 mt-0.5 bg-amber-500/10 text-amber-600">
              <Wrench className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium">Tool Result</span>
                {turn.tool_name && <Badge variant="outline" className="text-[10px]">{turn.tool_name}</Badge>}
                {turn.timestamp && <span className="text-xs text-muted-foreground">{turn.timestamp}</span>}
              </div>
              <pre className="text-xs text-muted-foreground whitespace-pre-wrap max-h-40 overflow-auto bg-muted/30 rounded p-2">
                {turn.content}
              </pre>
            </div>
          </div>
          {onDelete && (
            <button
              type="button"
              className="absolute top-2 right-2 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-muted transition-opacity"
              onClick={onDelete}
              title={t("chat.deleteTurn")}
            >
              <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive transition-colors" />
            </button>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn(isUser ? "border-l-4 border-l-primary" : "", "group relative")}>
      <CardContent className="pt-4">
        <div className="flex items-start gap-3">
          <div className={cn(
            "flex items-center justify-center w-7 h-7 rounded-full shrink-0 mt-0.5",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
          )}>
            {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium capitalize">{turn.role}</span>
              {turn.timestamp && <span className="text-xs text-muted-foreground">{turn.timestamp}</span>}
            </div>
            {reasoning && <ReasoningBlock content={reasoning} />}
            <MarkdownView content={turn.content} />
            <ToolCallsBlock calls={displayToolCalls} />
            <MediaBlock media={turn.media} sessionId={sessionId} />
            {!isUser && <ProviderDataPanel vendorExt={vendorExt} />}
          </div>
        </div>
        {onDelete && (
          <button
            type="button"
            className="absolute top-2 right-2 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-muted transition-opacity"
            onClick={onDelete}
            title={t("chat.deleteTurn")}
          >
            <Trash2 className="w-3.5 h-3.5 text-muted-foreground hover:text-destructive transition-colors" />
          </button>
        )}
      </CardContent>
    </Card>
  );
}

export default function SessionDetailPage() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const title = searchParams.get("title");
  const navigate = useNavigate();
  const { data: turns, isLoading } = useSessionTurns(id!);
  const deleteSession = useDeleteSession();
  const queryClient = useQueryClient();
  const [confirmOpen, setConfirmOpen] = useState(false);

  async function handleDelete() {
    await deleteSession.mutateAsync(id!);
    navigate("/admin/sessions");
  }

  // -- Turn deletion --
  const [turnToDelete, setTurnToDelete] = useState<number | null>(null);

  async function handleDeleteTurn() {
    if (turnToDelete == null || !id) return;
    try {
      await apiDelete(`/sessions/${encodeURIComponent(id)}/turns/${turnToDelete}`);
    } catch {
      // Continue even if API fails
    }
    queryClient.invalidateQueries({ queryKey: ["session-turns", id] });
    setTurnToDelete(null);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={() => navigate("/admin/sessions")}>
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <div className="flex-1 min-w-0">
          <h1 className="text-2xl font-bold truncate">{title || t("sessions.session")}</h1>
          <p className="text-xs text-muted-foreground font-mono truncate">{id}</p>
        </div>
        <Button variant="destructive" size="sm" onClick={() => setConfirmOpen(true)} disabled={deleteSession.isPending}>
          <Trash2 className="w-4 h-4 mr-1" />
          {t("common.delete")}
        </Button>
      </div>

      <ConfirmDialog
        open={confirmOpen}
        title={t("sessions.deleteSession")}
        description={t("sessions.deleteSessionDescription")}
        destructive
        onConfirm={() => { setConfirmOpen(false); handleDelete(); }}
        onCancel={() => setConfirmOpen(false)}
      />

      {isLoading && <p className="text-muted-foreground">{t("common.loading")}</p>}
      {turns && turns.length === 0 && <p className="text-muted-foreground">{t("sessions.noConversationTurns")}</p>}

      {turns && turns.length > 0 && (
        <div className="space-y-3">
          {(turns as TurnData[]).map((turn, i) => (
            <TurnCard
              key={i}
              turn={turn}
              sessionId={id}
              onDelete={turn.id != null ? () => setTurnToDelete(turn.id!) : undefined}
            />
          ))}
        </div>
      )}

      <ConfirmDialog
        open={turnToDelete !== null}
        title={t("chat.deleteTurn")}
        description={t("chat.deleteTurnConfirm")}
        destructive
        onConfirm={handleDeleteTurn}
        onCancel={() => setTurnToDelete(null)}
      />
    </div>
  );
}
