import { useState } from "react";
import { useParams, Link, useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft, Play, Loader2, Trash2, Clock, CheckCircle, XCircle,
  MessageSquare, CalendarClock, Timer, Repeat, Zap, Pencil,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StatusPill } from "@/components/status-pill";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { api } from "@/lib/api";
import { MarkdownView } from "@/components/markdown-view";
import { useRunTask, useDeleteTask, useUpdateTask } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import TaskForm, { taskFormDataFromTask } from "./form";

interface ScheduleInfo {
  type: "cron" | "one-shot" | "manual";
  expression?: string;
  summary?: string;
  run_at?: string;
  next_run?: string;
  time_until?: string;
  upcoming?: string[];
  status?: string;
  timezone?: string;
}

function ScheduleInline({ info }: { info: ScheduleInfo }) {
  const { t } = useTranslation();

  if (info.type === "manual") {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground border-t pt-3">
        <Zap className="w-4 h-4" />
        <span>{t("tasks.manualDescription")}</span>
      </div>
    );
  }

  if (info.type === "one-shot") {
    const isPast = info.status === "completed";
    const isOverdue = info.status === "overdue";
    return (
      <div className="flex items-center gap-2 flex-wrap text-sm border-t pt-3">
        <Timer className="w-4 h-4 text-muted-foreground" />
        <span className="font-mono text-xs">{info.run_at}</span>
        {isPast && <Badge variant="secondary">{t("tasks.completed")}</Badge>}
        {isOverdue && (
          <Badge className="bg-amber-500/20 text-amber-600 border-amber-500/30">
            {t("tasks.overdue")}
          </Badge>
        )}
        {info.time_until && (
          <span className="text-muted-foreground text-xs">
            {t("tasks.runsIn", { time: info.time_until })}
          </span>
        )}
        {info.timezone && (
          <Badge variant="outline" className="text-[10px] font-normal ml-auto">{info.timezone}</Badge>
        )}
      </div>
    );
  }

  // cron
  return (
    <div className="space-y-2 text-sm border-t pt-3">
      <div className="flex items-center gap-2 flex-wrap">
        <Repeat className="w-4 h-4 text-muted-foreground" />
        <code className="px-2 py-0.5 rounded bg-muted text-xs font-mono">{info.expression}</code>
        {info.summary && <span className="text-muted-foreground text-xs">{info.summary}</span>}
        {info.timezone && (
          <Badge variant="outline" className="text-[10px] font-normal ml-auto">{info.timezone}</Badge>
        )}
      </div>
      {info.next_run && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <CalendarClock className="w-3.5 h-3.5 shrink-0" />
          <span>{t("tasks.nextRun", { time: fmtDatetime(info.next_run) })}</span>
          {info.time_until && <Badge variant="secondary" className="text-[10px]">{info.time_until}</Badge>}
        </div>
      )}
    </div>
  );
}

function fmtDatetime(iso: string): string {
  try {
    const d = new Date(iso);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    const h = String(d.getHours()).padStart(2, "0");
    const min = String(d.getMinutes()).padStart(2, "0");
    return `${y}-${m}-${day} ${h}:${min}`;
  } catch {
    return iso;
  }
}

function fmtDuration(ms: number): string {
  if (ms > 60_000) return `${(ms / 60_000).toFixed(1)}m`;
  if (ms > 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${ms}ms`;
}

export default function TaskDetailPage() {
  const { t } = useTranslation();
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const runTask = useRunTask(name ?? "");
  const deleteTask = useDeleteTask(name ?? "");
  const updateTask = useUpdateTask(name ?? "");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [editing, setEditing] = useState(false);

  const task = useQuery({
    queryKey: ["task", name],
    queryFn: () => api<Record<string, unknown>>(`/tasks/${encodeURIComponent(name!)}`),
    enabled: !!name,
    refetchInterval: (query) => {
      const d = query.state.data as Record<string, unknown> | undefined;
      const live = d?.live_status as Record<string, unknown> | undefined;
      return live?.status === "running" ? 3_000 : false;
    },
  });

  const data = task.data as Record<string, unknown> | undefined;
  const liveStatus = data?.live_status as Record<string, unknown> | undefined;
  const isRunning = liveStatus?.status === "running";
  const scheduleInfo = data?.schedule_info as ScheduleInfo | undefined;

  if (task.isLoading) return <p className="text-muted-foreground">{t("common.loading")}</p>;
  if (!data) return <p className="text-muted-foreground">{t("tasks.taskNotFound")}</p>;

  const runs = (data.runs ?? []) as Array<Record<string, unknown>>;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/admin/tasks">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-4 h-4" />
          </Button>
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold">{name}</h1>
          <div className="flex items-center gap-2 mt-1">
            {!!data.task_status && (
              <StatusPill status={String(data.task_status)} />
            )}
            {scheduleInfo && (
              <Badge variant="outline" className="text-xs">
                {scheduleInfo.type === "cron"
                  ? scheduleInfo.summary ?? "cron"
                  : scheduleInfo.type === "one-shot"
                    ? t("tasks.oneShot").toLowerCase()
                    : t("tasks.manual").toLowerCase()}
              </Badge>
            )}
          </div>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setEditing(!editing)}
        >
          <Pencil className="w-4 h-4 mr-1" />
          {editing ? t("tasks.cancelEdit") : t("tasks.editTask")}
        </Button>
        <Button
          size="sm"
          onClick={() => runTask.mutate()}
          disabled={runTask.isPending || isRunning}
        >
          {runTask.isPending || isRunning ? (
            <Loader2 className="w-4 h-4 mr-1 animate-spin" />
          ) : (
            <Play className="w-4 h-4 mr-1" />
          )}
          {runTask.isPending ? t("tasks.starting") : isRunning ? t("tasks.running") : t("common.run")}
        </Button>
        <Button
          size="sm"
          variant="destructive"
          onClick={() => setConfirmDelete(true)}
          disabled={deleteTask.isPending}
        >
          <Trash2 className="w-4 h-4 mr-1" />
          {t("common.delete")}
        </Button>
      </div>

      <ConfirmDialog
        open={confirmDelete}
        title={t("tasks.deleteTask")}
        description={t("tasks.deleteTaskDescription", { name: name ?? "" })}
        destructive
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => {
          setConfirmDelete(false);
          deleteTask.mutate(undefined, {
            onSuccess: () => navigate("/admin/tasks"),
          });
        }}
      />

      {/* Running indicator */}
      {isRunning && (
        <div className="flex items-center gap-3 p-3 rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-400 text-sm animate-pulse">
          <Loader2 className="w-4 h-4 animate-spin shrink-0" />
          <span className="font-medium">{t("tasks.taskRunning")}</span>
          {!!liveStatus?.started_at && (
            <span className="text-xs opacity-75 ml-auto">
              {t("tasks.started")}: {String(liveStatus.started_at)}
            </span>
          )}
        </div>
      )}

      {/* Run feedback banner */}
      {runTask.isSuccess && !isRunning && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
          <CheckCircle className="w-4 h-4 shrink-0" />
          {t("tasks.taskStartedSuccess")}
        </div>
      )}
      {runTask.isError && (
        <div className="flex items-center gap-2 p-3 rounded-md bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-sm">
          <XCircle className="w-4 h-4 shrink-0" />
          {t("tasks.taskStartFailed", { error: runTask.error?.message ?? "Unknown error" })}
        </div>
      )}

      {/* Live status & schedule */}
      {(liveStatus || scheduleInfo) && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              {t("tasks.liveStatus")}
              {isRunning && <Loader2 className="w-4 h-4 animate-spin text-amber-500" />}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {liveStatus && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground text-xs">{t("tasks.state")}</p>
                  <StatusPill status={String(liveStatus.status ?? "unknown")} />
                </div>
                {!!liveStatus.started_at && (
                  <div>
                    <p className="text-muted-foreground text-xs">{t("tasks.started")}</p>
                    <p>{String(liveStatus.started_at)}</p>
                  </div>
                )}
                {!!liveStatus.finished_at && (
                  <div>
                    <p className="text-muted-foreground text-xs">{t("tasks.finished")}</p>
                    <p>{String(liveStatus.finished_at)}</p>
                  </div>
                )}
                {!!liveStatus.error && (
                  <div className="col-span-full">
                    <p className="text-muted-foreground text-xs">{t("tasks.error")}</p>
                    <p className="text-red-500 text-xs">{String(liveStatus.error)}</p>
                  </div>
                )}
              </div>
            )}
            {scheduleInfo && (
              <ScheduleInline info={scheduleInfo} />
            )}
          </CardContent>
        </Card>
      )}

      {/* Definition / Edit form */}
      {editing ? (
        <>
          {updateTask.isSuccess && (
            <div className="flex items-center gap-2 p-3 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-sm">
              <CheckCircle className="w-4 h-4 shrink-0" />
              {t("tasks.updateSuccess")}
            </div>
          )}
          <TaskForm
            mode="edit"
            initialData={taskFormDataFromTask(data)}
            onSubmit={(payload) => {
              updateTask.mutate(payload, {
                onSuccess: () => { setEditing(false); task.refetch(); },
              });
            }}
            isPending={updateTask.isPending}
            error={updateTask.isError ? t("tasks.updateError", { error: updateTask.error?.message ?? "Unknown" }) : undefined}
          />
        </>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t("tasks.definition")}</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
              {!!data.prompt && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.prompt")}</dt>
                  <dd><MarkdownView content={String(data.prompt)} /></dd>
                </>
              )}
              {!!data.description && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.description")}</dt>
                  <dd>{String(data.description)}</dd>
                </>
              )}
              {!!data.agent && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.taskAgent")}</dt>
                  <dd><Badge variant="outline">{String(data.agent)}</Badge></dd>
                </>
              )}
              {!!data.context && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.taskContext")}</dt>
                  <dd className="font-mono text-xs break-all">{String(data.context)}</dd>
                </>
              )}
              {!!(data.skills && (data.skills as string[]).length > 0) && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.taskSkill")}</dt>
                  <dd><Badge variant="outline">{(data.skills as string[])[0]}</Badge></dd>
                </>
              )}
              {!!data.workflow && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.taskWorkflow")}</dt>
                  <dd><Badge variant="outline">{String(data.workflow)}</Badge></dd>
                </>
              )}
              {!!data.created_at && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.created")}</dt>
                  <dd>{String(data.created_at)}</dd>
                </>
              )}
              {!!data.updated_at && (
                <>
                  <dt className="text-muted-foreground">{t("tasks.updated")}</dt>
                  <dd>{String(data.updated_at)}</dd>
                </>
              )}
            </dl>
          </CardContent>
        </Card>
      )}

      {/* Run history */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center justify-between">
            {t("tasks.runHistory")}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => task.refetch()}
              disabled={task.isFetching}
            >
              {task.isFetching ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                t("common.refresh")
              )}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {runs.length === 0 ? (
            <p className="text-muted-foreground text-sm">{t("tasks.noRuns")}</p>
          ) : (
            <div className="space-y-1">
              {runs.map((r, i) => {
                const runSessionId = r.session_id as string | undefined;
                return (
                  <div
                    key={i}
                    className={
                      "flex items-center gap-3 text-sm border-b border-border pb-2 last:border-0 py-1.5 rounded-sm " +
                      (runSessionId
                        ? "hover:bg-accent/50 cursor-pointer transition-colors px-2 -mx-2"
                        : "")
                    }
                    onClick={
                      runSessionId
                        ? () =>
                            navigate(
                              `/admin/sessions/${encodeURIComponent(runSessionId)}?title=${encodeURIComponent(name ?? "")} [${String(r.execution_id)}]`,
                            )
                        : undefined
                    }
                  >
                    <Clock className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                    <span className="text-muted-foreground flex-1">
                      {String(r.started_at ?? r.timestamp ?? "")}
                    </span>
                    {!!r.execution_id && (
                      <span className="text-xs text-muted-foreground font-mono">
                        {String(r.execution_id)}
                      </span>
                    )}
                    {r.duration_ms != null && (
                      <span className="text-xs text-muted-foreground">
                        {fmtDuration(Number(r.duration_ms))}
                      </span>
                    )}
                    <StatusPill status={String(r.status ?? "unknown")} />
                    {runSessionId && (
                      <MessageSquare className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
