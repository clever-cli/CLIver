import { useState } from "react";
import { Plus, Trash2, Bot, Pencil } from "lucide-react";
import {
  useModels, useCreateModel, useUpdateModel, useDeleteModel,
  useProviders, useCreateProvider,
  useKeys, useCreateKey,
  type ModelInfo, type ModelProvider,
} from "@/hooks/use-api";
import { useTranslation } from "@/i18n";
import { ModelFormSchema, type ModelFormData } from "@/lib/schemas";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/confirm-dialog";

const PROVIDER_COLORS: Record<string, string> = {
  openai: "bg-amber-500",
  anthropic: "bg-orange-500",
};

const _MODALITY_COLORS: Record<string, string> = {
  text: "bg-emerald-500",
  image: "bg-purple-500",
  audio: "bg-red-500",
  video: "bg-pink-500",
};

const CATEGORIES = ["Text", "Image", "Audio", "Video"] as const;

const CATEGORY_COLORS: Record<string, string> = {
  Text: "bg-emerald-500",
  Image: "bg-purple-500",
  Audio: "bg-red-500",
  Video: "bg-pink-500",
};

function categoryKey(cat: string): string {
  const m: Record<string, string> = {
    text: "models.categoryText", Text: "models.categoryText",
    image: "models.categoryImage", Image: "models.categoryImage",
    audio: "models.categoryAudio", Audio: "models.categoryAudio",
    video: "models.categoryVideo", Video: "models.categoryVideo",
  };
  return m[cat] || cat;
}

function tabLabelKey(tab: string): string {
  const m: Record<string, string> = {
    Text: "models.textModels", Image: "models.imageModels",
    Audio: "models.audioModels", Video: "models.videoModels",
  };
  return m[tab] || `models.${tab}Models`;
}

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMin = Math.floor((now - then) / 60000);
  if (diffMin < 1) return "Just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return `${Math.floor(diffHr / 24)}d ago`;
}

function OptionsTable({ options, onChange, t }: {
  options: Record<string, string>;
  onChange: (o: Record<string, string>) => void;
  t: (key: string) => string;
}) {
  const entries = Object.entries(options);
  const set = (k: string, v: string) => onChange({ ...options, [k]: v });
  const remove = (k: string) => {
    const copy = { ...options };
    delete copy[k];
    onChange(copy);
  };
  const add = () => onChange({ ...options, "": "" });

  return (
    <div className="space-y-1">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted-foreground">
            <th className="text-left font-medium py-1">{t("models.keyColumn")}</th>
            <th className="text-left font-medium py-1">{t("models.valueColumn")}</th>
            <th className="w-8"></th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, v], i) => (
            <tr key={i}>
              <td className="pr-1 py-0.5">
                <Input
                  value={k}
                  onChange={(e) => { remove(k); set(e.target.value, v); }}
                  className="h-7 text-xs"
                  placeholder={t("models.keyPlaceholder")}
                />
              </td>
              <td className="pr-1 py-0.5">
                <Input
                  value={v}
                  onChange={(e) => set(k, e.target.value)}
                  className="h-7 text-xs"
                  placeholder={t("models.valuePlaceholder")}
                />
              </td>
              <td className="py-0.5">
                <button onClick={() => remove(k)} className="text-destructive hover:text-destructive/80">
                  <Trash2 className="w-3 h-3" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Button variant="ghost" size="sm" className="text-xs h-6" onClick={add}>
        + {t("models.addOption")}
      </Button>
    </div>
  );
}

function ModelForm({ model, onChange, providers, t }: {
  model: Partial<ModelInfo>;
  onChange: (m: Partial<ModelInfo>) => void;
  providers: ModelProvider[];
  t: (key: string, params?: Record<string, string>) => string;
}) {
  const set = (k: string, v: unknown) => onChange({ ...model, [k]: v });

  const handleOptionsChange = (opts: Record<string, string>) => {
    const cleaned: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(opts)) {
      const tk = k.trim();
      if (tk) {
        const num = Number(v);
        cleaned[tk] = isNaN(num) ? v : num;
      } else {
        cleaned[k] = v;
      }
    }
    set("options", cleaned);
  };

  const optionsStr: Record<string, string> = {};
  if (model.options) {
    for (const [k, v] of Object.entries(model.options)) {
      optionsStr[k] = String(v);
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium">{t("models.providerLabel")} *</label>
        <select
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={model.provider || ""}
          onChange={(e) => set("provider", e.target.value)}
          autoFocus
        >
          <option value="" disabled>{t("models.providerLabel")}</option>
          {providers.map((p) => (
            <option key={p.id} value={p.id}>{p.name} ({p.type})</option>
          ))}
        </select>
      </div>
      <div>
        <label className="text-sm font-medium">{t("models.nameLabel")} *</label>
        <Input
          value={model.name || ""}
          onChange={(e) => set("name", e.target.value)}
          placeholder={t("models.namePlaceholder")}
        />
      </div>
      <div>
        <label className="text-sm font-medium">{t("models.category")} *</label>
        <select
          className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={model.category || "Text"}
          onChange={(e) => set("category", e.target.value)}
        >
          {CATEGORIES.map((cat) => (
            <option key={cat} value={cat}>{t(categoryKey(cat))}</option>
          ))}
        </select>
      </div>
      <div>
        <label className="text-sm font-medium">{t("models.apiModelName")} *</label>
        <Input
          value={model.model || ""}
          onChange={(e) => set("model", e.target.value)}
          placeholder={t("models.apiModelNamePlaceholder")}
        />
      </div>
      <div>
        <label className="text-sm font-medium">{t("models.apiUrl")}</label>
        <Input
          value={model.api_url || ""}
          onChange={(e) => set("api_url", e.target.value)}
          placeholder="https://api.example.com/v1/generate"
        />
      </div>
      <div>
        <label className="text-sm font-medium">{t("models.optionsLabel")}</label>
        <OptionsTable options={optionsStr} onChange={handleOptionsChange} t={t} />
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<string>("all");

  const { data: models, isLoading } = useModels();

  const filteredModels = (models || []).filter((m) => {
    if (activeTab === "all") return true;
    const raw = (m.category || "text").toLowerCase();
    return raw === activeTab.toLowerCase();
  });

  const categoryCounts: Record<string, number> = {};
  for (const m of (models || [])) {
    const raw = m.category || "text";
    const cat = raw.charAt(0).toUpperCase() + raw.slice(1);
    categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
  }
  const { data: providers } = useProviders();
  const { data: keys } = useKeys();
  const createModel = useCreateModel();
  const [editTarget, setEditTarget] = useState<ModelInfo | null>(null);
  const updateModel = useUpdateModel(editTarget?.id || "");
  const deleteModel = useDeleteModel();
  const createProvider = useCreateProvider();
  const createKey = useCreateKey();

  const [showCreate, setShowCreate] = useState(false);
  const [showProviderDialog, setShowProviderDialog] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<ModelInfo | null>(null);
  const [form, setForm] = useState<Partial<ModelInfo>>({ options: {} });
  const [showNewKey, setShowNewKey] = useState(false);
  const [providerForm, setProviderForm] = useState({
    name: "", type: "openai", api_key: "", api_url: "",
    rate_requests: "", rate_period: "1m", rate_margin: "0.1",
    pricing_currency: "USD", pricing_input: "", pricing_output: "", pricing_cached_input: "",
  });
  const [keyForm, setKeyForm] = useState({ name: "", value: "", description: "" });

  const resetForm = () => {
    setForm({ options: {} });
  };

  const cleanOptions = (opts?: Record<string, unknown>) => {
    if (!opts) return {};
    const cleaned: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(opts)) {
      if (k.trim()) cleaned[k] = v;
    }
    return cleaned;
  };

  const resetProviderForm = () => {
    setProviderForm({
      name: "", type: "openai", api_key: "", api_url: "",
      rate_requests: "", rate_period: "1m", rate_margin: "0.1",
      pricing_currency: "USD", pricing_input: "", pricing_output: "", pricing_cached_input: "",
    });
  };

  const buildProviderPayload = () => {
    const payload: Record<string, unknown> = {
      name: providerForm.name.trim(),
      type: providerForm.type,
      api_url: providerForm.api_url || undefined,
      api_key: providerForm.api_key || undefined,
    };
    if (providerForm.rate_requests) {
      payload.rate_limit = {
        requests: Number(providerForm.rate_requests),
        period: providerForm.rate_period,
        margin: Number(providerForm.rate_margin),
      };
    }
    const pi = providerForm.pricing_input;
    const po = providerForm.pricing_output;
    if (pi || po) {
      payload.pricing = { currency: providerForm.pricing_currency };
      if (pi) (payload.pricing as Record<string, unknown>).input = Number(pi);
      if (po) (payload.pricing as Record<string, unknown>).output = Number(po);
      if (providerForm.pricing_cached_input)
        (payload.pricing as Record<string, unknown>).cached_input = Number(providerForm.pricing_cached_input);
    }
    return payload;
  };

  const handleCreate = async () => {
    const result = ModelFormSchema.safeParse({
      name: form.name?.trim(),
      provider: form.provider,
      category: form.category || "Text",
      model: form.model || form.name?.trim() || "",
      api_url: form.api_url || null,
      options: cleanOptions(form.options),
    });
    if (!result.success) {
      // Surface validation errors — in production, show these in the form UI
      console.error("Form validation failed:", result.error.flatten().fieldErrors);
      return;
    }
    const data: ModelFormData = result.data;
    createModel.mutate({
      name: data.name,
      provider: data.provider,
      model: data.model,
      category: data.category,
      api_url: data.api_url ?? null,
      options: data.options ? cleanOptions(data.options as Record<string, unknown>) : {},
    }, {
      onSuccess: () => {
        setShowCreate(false);
        resetForm();
      },
    });
  };

  const handleUpdate = async () => {
    if (!editTarget || !form.name?.trim()) return;
    updateModel.mutate({
      name: form.name.trim(),
      model: form.model || form.name.trim(),
      category: form.category || "Text",
      api_url: form.api_url || null,
      options: cleanOptions(form.options),
    }, {
      onSuccess: () => {
        setEditTarget(null);
        resetForm();
      },
    });
  };

  const openEdit = (m: ModelInfo) => {
    const raw = m.category || "text";
    setEditTarget(m);
    setForm({ ...m, category: raw.charAt(0).toUpperCase() + raw.slice(1) });
  };

  const handleCreateKey = async () => {
    if (!keyForm.name.trim() || !keyForm.value.trim()) return;
    await createKey.mutateAsync({
      name: keyForm.name.trim(),
      value: keyForm.value,
      description: keyForm.description,
    });
    setProviderForm({ ...providerForm, api_key: `{{ key('${keyForm.name.trim()}') }}` });
    setShowNewKey(false);
    setKeyForm({ name: "", value: "", description: "" });
  };

  const handleCreateProviderStandalone = async () => {
    if (!providerForm.name.trim()) return;
    await createProvider.mutateAsync(buildProviderPayload());
    setShowProviderDialog(false);
    resetProviderForm();
  };

  const tabs = ["all", "Text", "Image", "Audio", "Video"];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">{t("models.title")}</h1>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={() => {
            resetProviderForm();
            setShowNewKey(false);
            setShowProviderDialog(true);
          }}>
            <Plus className="w-4 h-4 mr-1" />
            {t("models.newProviderButton")}
          </Button>
          <Button onClick={() => { resetForm(); setShowCreate(true); }}>
            <Plus className="w-4 h-4 mr-1" />
            {t("models.newModel")}
          </Button>
        </div>
      </div>

      {isLoading && <p className="text-sm text-muted-foreground">{t("common.loading")}</p>}

      {!isLoading && models && models.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Bot className="w-12 h-12 text-muted-foreground/50 mb-4" />
          <h2 className="text-lg font-medium">{t("models.noModels")}</h2>
          <p className="text-sm text-muted-foreground mt-1 max-w-sm">{t("models.noModelsDesc")}</p>
          <Button className="mt-4" onClick={() => { resetForm(); setShowCreate(true); }}>
            <Plus className="w-4 h-4 mr-1" />
            {t("models.createTitle")}
          </Button>
        </div>
      )}

      {/* Type filter tabs */}
      {models && models.length > 0 && (
      <div className="flex gap-2">
        {tabs.map((tab) => {
          const count = tab === "all"
            ? (models || []).length
            : (categoryCounts[tab] || 0);
          return (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                activeTab === tab
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/80"
              }`}
            >
              {tab === "all" ? t("models.allModels") : t(tabLabelKey(tab))}
              {models && ` (${count})`}
            </button>
          );
        })}
      </div>
      )}

      {/* Empty state for filtered category */}
      {!isLoading && models && models.length > 0 && filteredModels.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <Bot className="w-12 h-12 text-muted-foreground/50 mb-4" />
          <h2 className="text-lg font-medium">{t("models.noModels")}</h2>
        </div>
      )}

      {/* Card grid */}
      {filteredModels.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map((model) => {
          const provider = (providers || []).find((p) => p.id === model.provider);
          const providerName = provider?.name || model.provider;
          const cat = model.category || "Text";
          return (
            <Card
              key={model.id}
              className="p-4 hover:shadow-md transition-shadow cursor-pointer group"
              onClick={() => openEdit(model)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium truncate">{model.name}</h3>
                  <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
                    <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded text-white ${PROVIDER_COLORS[provider?.type || ""] || "bg-gray-500"}`}>
                      {providerName}
                    </span>
                    <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded text-white ${CATEGORY_COLORS[cat] || "bg-gray-400"}`}>
                      {t(categoryKey(cat))}
                    </span>
                    {model.is_default === 1 && (
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-primary/20 text-primary">default</span>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between mt-3 pt-3 border-t">
                <span className="text-xs text-muted-foreground">{timeAgo(model.updated_at || "")}</span>
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="p-1 hover:bg-muted rounded" onClick={(e) => { e.stopPropagation(); openEdit(model); }}>
                    <Pencil className="w-3.5 h-3.5" />
                  </button>
                  <button className="p-1 hover:bg-destructive/10 rounded text-destructive" onClick={(e) => { e.stopPropagation(); setDeleteTarget(model); }}>
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </Card>
          );
        })}
      </div>
      )}

      {/* Create Dialog */}
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("models.createTitle")}</DialogTitle>
          </DialogHeader>

          <ModelForm model={form} onChange={setForm} providers={providers || []} t={t} />

          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setShowCreate(false)}>{t("common.cancel")}</Button>
            <Button onClick={handleCreate} disabled={!form.name?.trim() || !form.provider || createModel.isPending}>
              {createModel.isPending ? t("models.saving") : t("common.save")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={!!editTarget} onOpenChange={() => setEditTarget(null)}>
        <DialogContent className="max-w-lg max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("models.editTitle")}</DialogTitle>
          </DialogHeader>
          <ModelForm model={form} onChange={setForm} providers={providers || []} t={t} />
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => setEditTarget(null)}>{t("common.cancel")}</Button>
            <Button onClick={handleUpdate} disabled={!form.name?.trim() || updateModel.isPending}>
              {updateModel.isPending ? t("models.saving") : t("common.save")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={!!deleteTarget}
        onCancel={() => setDeleteTarget(null)}
        title={t("models.deleteTitle")}
        description={t("models.deleteDesc", { name: deleteTarget ? `${deleteTarget.name}` : "" })}
        onConfirm={() => {
          if (deleteTarget) {
            deleteModel.mutate(deleteTarget.id);
            setDeleteTarget(null);
          }
        }}
      />

      {/* New Provider Dialog */}
      <Dialog open={showProviderDialog} onOpenChange={setShowProviderDialog}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t("models.newProviderTitle")}</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">{t("models.providerName")}</label>
              <Input
                value={providerForm.name}
                onChange={(e) => setProviderForm({ ...providerForm, name: e.target.value })}
                placeholder={t("models.providerName")}
              />
            </div>
            <div>
              <label className="text-sm font-medium">{t("models.providerType")}</label>
              <select
                className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={providerForm.type}
                onChange={(e) => setProviderForm({ ...providerForm, type: e.target.value })}
              >
                <option value="openai">openai</option>
                <option value="anthropic">anthropic</option>
              </select>
            </div>
            <div>
              <label className="text-sm font-medium">{t("models.providerApiUrl")}</label>
              <Input
                value={providerForm.api_url}
                onChange={(e) => setProviderForm({ ...providerForm, api_url: e.target.value })}
                placeholder={t("models.providerApiUrlPlaceholder")}
              />
            </div>
            <div>
              <label className="text-sm font-medium">{t("models.providerApiKey")}</label>
              <div className="flex gap-1">
                <select
                  className="flex-1 rounded-md border border-input bg-background px-2 py-1.5 text-sm"
                  value={providerForm.api_key}
                  onChange={(e) => {
                    if (e.target.value === "__new__") { setShowNewKey(true); return; }
                    setProviderForm({ ...providerForm, api_key: e.target.value });
                  }}
                >
                  <option value="">{t("models.selectKey")}</option>
                  {(keys || []).map((k) => (
                    <option key={k.name} value={`{{ key('${k.name}') }}`}>🔑 {k.name}</option>
                  ))}
                  <option value="__new__">+ {t("models.createKey")}</option>
                </select>
              </div>
              {showNewKey && (
                <div className="mt-2 space-y-1.5 border border-green-500/30 rounded-lg p-2 bg-green-500/5">
                  <Input
                    value={keyForm.name}
                    onChange={(e) => setKeyForm({ ...keyForm, name: e.target.value })}
                    placeholder={t("models.keyName")}
                    className="h-7 text-xs"
                  />
                  <Input
                    value={keyForm.value}
                    onChange={(e) => setKeyForm({ ...keyForm, value: e.target.value })}
                    placeholder={t("models.keyValue")}
                    className="h-7 text-xs"
                  />
                  <Input
                    value={keyForm.description}
                    onChange={(e) => setKeyForm({ ...keyForm, description: e.target.value })}
                    placeholder={t("models.keyDescription")}
                    className="h-7 text-xs"
                  />
                  <div className="flex justify-end gap-1">
                    <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => setShowNewKey(false)}>
                      {t("common.cancel")}
                    </Button>
                    <Button size="sm" className="h-6 text-xs" onClick={handleCreateKey} disabled={createKey.isPending}>
                      {t("models.createAndSelect")}
                    </Button>
                  </div>
                </div>
              )}
            </div>

            {/* Rate Limit */}
            <div>
              <label className="text-sm font-medium">{t("models.rateLimitLabel")}</label>
              <div className="grid grid-cols-3 gap-2 mt-1">
                <div>
                  <label className="text-[10px] text-muted-foreground">{t("models.rateLimitRequests")}</label>
                  <Input
                    type="number"
                    value={providerForm.rate_requests}
                    onChange={(e) => setProviderForm({ ...providerForm, rate_requests: e.target.value })}
                    className="h-8 text-xs"
                    placeholder="60"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t("models.rateLimitPeriod")}</label>
                  <Input
                    value={providerForm.rate_period}
                    onChange={(e) => setProviderForm({ ...providerForm, rate_period: e.target.value })}
                    className="h-8 text-xs"
                    placeholder="1m"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t("models.rateLimitMargin")}</label>
                  <Input
                    type="number"
                    step="0.1"
                    value={providerForm.rate_margin}
                    onChange={(e) => setProviderForm({ ...providerForm, rate_margin: e.target.value })}
                    className="h-8 text-xs"
                    placeholder="0.1"
                  />
                </div>
              </div>
            </div>

            {/* Pricing */}
            <div>
              <label className="text-sm font-medium">{t("models.pricing")}</label>
              <div className="grid grid-cols-4 gap-2 mt-1">
                <div>
                  <label className="text-[10px] text-muted-foreground">{t("models.pricingCurrency")}</label>
                  <Input
                    value={providerForm.pricing_currency}
                    onChange={(e) => setProviderForm({ ...providerForm, pricing_currency: e.target.value })}
                    className="h-8 text-xs"
                    placeholder="USD"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t("models.pricingInput")}</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={providerForm.pricing_input}
                    onChange={(e) => setProviderForm({ ...providerForm, pricing_input: e.target.value })}
                    className="h-8 text-xs"
                    placeholder="0"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t("models.pricingOutput")}</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={providerForm.pricing_output}
                    onChange={(e) => setProviderForm({ ...providerForm, pricing_output: e.target.value })}
                    className="h-8 text-xs"
                    placeholder="0"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t("models.pricingCachedInput")}</label>
                  <Input
                    type="number"
                    step="0.01"
                    value={providerForm.pricing_cached_input}
                    onChange={(e) => setProviderForm({ ...providerForm, pricing_cached_input: e.target.value })}
                    className="h-8 text-xs"
                    placeholder="0"
                  />
                </div>
              </div>
            </div>
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={() => {
              setShowProviderDialog(false);
              setShowNewKey(false);
            }}>
              {t("common.cancel")}
            </Button>
            <Button onClick={handleCreateProviderStandalone} disabled={!providerForm.name.trim() || createProvider.isPending}>
              {createProvider.isPending ? t("models.saving") : t("common.save")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
