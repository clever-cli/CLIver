import { useState, useEffect, useRef, useCallback } from "react";
import { useModels, useSkills, useMCPServers } from "@/hooks/use-api";
import { useTranslation } from "@/i18n";

interface LabConfigPanelProps {
  selectedModel: string;
  selectedImageModel: string;
  selectedAudioModel: string;
  selectedVideoModel: string;
  systemPrompt: string;
  selectedSkills: string[];
  selectedMCPServerIds: string[];
  onModelChange: (v: string) => void;
  onImageModelChange: (v: string) => void;
  onAudioModelChange: (v: string) => void;
  onVideoModelChange: (v: string) => void;
  onSystemPromptChange: (v: string) => void;
  onSkillsChange: (v: string[]) => void;
  onMCPServersChange: (v: string[]) => void;
  onSave: () => void;
  saving: boolean;
}

export function LabConfigPanel({
  selectedModel,
  selectedImageModel,
  selectedAudioModel,
  selectedVideoModel,
  systemPrompt,
  selectedSkills,
  selectedMCPServerIds,
  onModelChange,
  onImageModelChange,
  onAudioModelChange,
  onVideoModelChange,
  onSystemPromptChange,
  onSkillsChange,
  onMCPServersChange,
  onSave,
  saving,
}: LabConfigPanelProps) {
  const { t } = useTranslation();
  const { data: modelsData } = useModels();
  const { data: imageModelsData } = useModels("Image");
  const { data: audioModelsData } = useModels("Audio");
  const { data: videoModelsData } = useModels("Video");
  const { data: skills } = useSkills();
  const { data: mcpServers } = useMCPServers();

  const [localSysMsg, setLocalSysMsg] = useState(systemPrompt);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setLocalSysMsg(systemPrompt);
  }, [systemPrompt]);

  const handleSysMsgChange = useCallback((v: string) => {
    setLocalSysMsg(v);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => onSystemPromptChange(v), 400);
  }, [onSystemPromptChange]);

  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  const BUILTIN_SKILLS = ["brainstorm", "write-plan", "execute-plan"];

  const modelList: string[] = Array.isArray(modelsData) ? modelsData.map((m) => m.name) : [];
  const imageModelList: string[] = Array.isArray(imageModelsData) ? imageModelsData.map((m) => m.name) : [];
  const audioModelList: string[] = Array.isArray(audioModelsData) ? audioModelsData.map((m) => m.name) : [];
  const videoModelList: string[] = Array.isArray(videoModelsData) ? videoModelsData.map((m) => m.name) : [];
  const allSkills: string[] = skills
    ? skills.map((s) => String((s as Record<string, unknown>).name ?? "")).filter(Boolean)
    : [];
  const builtinList = allSkills.filter((s) => BUILTIN_SKILLS.includes(s));
  const optionalList = allSkills.filter((s) => !BUILTIN_SKILLS.includes(s));

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto space-y-4 p-3">
        {/* Text Model */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.textModel")}
          </label>
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="">{t("lab.modelPlaceholder")}</option>
            {modelList.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        {/* Image Model */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.imageModel")}
          </label>
          <select
            value={selectedImageModel}
            onChange={(e) => onImageModelChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="">{t("lab.modelPlaceholder")}</option>
            {imageModelList.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        {/* Audio Model */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.audioModel")}
          </label>
          <select
            value={selectedAudioModel}
            onChange={(e) => onAudioModelChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="">{t("lab.modelPlaceholder")}</option>
            {audioModelList.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        {/* Video Model */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.videoModel")}
          </label>
          <select
            value={selectedVideoModel}
            onChange={(e) => onVideoModelChange(e.target.value)}
            className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="">{t("lab.modelPlaceholder")}</option>
            {videoModelList.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        {/* System Prompt */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.systemPrompt")}
          </label>
          <textarea
            value={localSysMsg}
            onChange={(e) => handleSysMsgChange(e.target.value)}
            rows={4}
            className="mt-1 w-full rounded-md border border-input bg-background px-2 py-1.5 text-xs resize-y focus:outline-none focus:ring-2 focus:ring-ring"
            placeholder={t("lab.systemPromptPlaceholder")}
          />
        </div>

        {/* MCP Servers */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.mcpServers")}
          </label>
          {!mcpServers || mcpServers.length === 0 ? (
            <p className="text-[11px] text-muted-foreground mt-1">{t("mcpServers.noServers")}</p>
          ) : (
            <div className="mt-1 flex flex-wrap gap-1">
              {mcpServers.map((srv) => {
                const active = selectedMCPServerIds.includes(srv.id);
                return (
                  <button
                    key={srv.id}
                    type="button"
                    onClick={() =>
                      onMCPServersChange(
                        active
                          ? selectedMCPServerIds.filter((id) => id !== srv.id)
                          : [...selectedMCPServerIds, srv.id],
                      )
                    }
                    className={`rounded-md px-2 py-0.5 text-[11px] transition-colors ${
                      active
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground hover:bg-secondary"
                    }`}
                  >
                    {srv.name}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Knowledge Bases (placeholder) */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.knowledgeBases")}
          </label>
          <p className="text-[11px] text-muted-foreground mt-1">{t("lab.comingSoon")}</p>
        </div>

        {/* Skills */}
        <div>
          <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
            {t("lab.skills")}
          </label>
          {allSkills.length === 0 ? (
            <p className="text-[11px] text-muted-foreground mt-1">{t("common.loading")}</p>
          ) : (
            <div className="mt-1 space-y-2">
              {/* Builtin skills — always on, non-removable */}
              {builtinList.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {builtinList.map((s) => (
                    <span
                      key={s}
                      className="rounded-md px-2 py-0.5 text-[11px] bg-blue-100 text-blue-700 border border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              )}
              {/* Optional skills — toggleable */}
              {optionalList.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {optionalList.map((s) => {
                    const active = selectedSkills.includes(s);
                    return (
                      <button
                        key={s}
                        type="button"
                        onClick={() =>
                          onSkillsChange(
                            active ? selectedSkills.filter((x) => x !== s) : [...selectedSkills, s],
                          )
                        }
                        className={`rounded-md px-2 py-0.5 text-[11px] transition-colors ${
                          active
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted text-muted-foreground hover:bg-secondary"
                        }`}
                      >
                        {s}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Save button */}
      <div className="p-3 border-t shrink-0">
        <button
          type="button"
          onClick={onSave}
          disabled={saving}
          className="w-full rounded-md bg-primary text-primary-foreground px-3 py-1.5 text-xs font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {saving ? t("lab.configSaved") : t("lab.saveConfig")}
        </button>
      </div>
    </div>
  );
}
