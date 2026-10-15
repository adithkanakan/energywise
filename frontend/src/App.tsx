import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface ForecastPoint {
  timestamp: string;
  predicted_kwh: number;
}

interface ForecastResponse {
  building: string;
  horizon_hours: number;
  unit: string;
  forecast: ForecastPoint[];
}

interface MetadataResponse {
  model_version: string;
  training_range: { start: string; end: string };
  target_unit: string;
  horizon_hours: number;
  evaluation_note: string;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-IN", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}
function formatChartLabel(iso: string): string {
  return new Date(iso).toLocaleString("en-IN", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}
function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-IN", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface ChartPoint {
  timestamp: string;
  label: string;
  actual: number | null;
  forecast: number | null;
}

function DemandChart({ building }: { building: string }) {
  const [data, setData] = useState<ChartPoint[]>([]);
  const [boundaryLabel, setBoundaryLabel] = useState<string | null>(null);
  const [chartError, setChartError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [histRes, forecastRes] = await Promise.all([
          fetch(`${API_URL}/history?building=${building}&days=3`),
          fetch(`${API_URL}/forecast`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ building, horizon_hours: 24 }),
          }),
        ]);

        if (!histRes.ok || !forecastRes.ok) throw new Error("fetch failed");

        const hist: { history: { timestamp: string; actual_kwh: number | null }[] } =
          await histRes.json();
        const fc: ForecastResponse = await forecastRes.json();

        const historyPoints: ChartPoint[] = hist.history.map((h) => ({
          timestamp: h.timestamp,
          label: formatChartLabel(h.timestamp),
          actual: h.actual_kwh,
          forecast: null,
        }));

        const forecastPoints: ChartPoint[] = fc.forecast.map((f) => ({
          timestamp: f.timestamp,
          label: formatChartLabel(f.timestamp),
          actual: null,
          forecast: f.predicted_kwh,
        }));

        if (historyPoints.length > 0) {
          setBoundaryLabel(historyPoints[historyPoints.length - 1].label);
        }

        setData([...historyPoints, ...forecastPoints]);
      } catch {
        setChartError("Could not load chart data");
      }
    }
    load();
  }, [building]);

  if (chartError) {
    return <p className="text-red-600">{chartError}</p>;
  }

  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-sm text-slate-500 mb-2">
        Actual vs forecast demand — {building}
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11 }} interval={5} />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "kWh", angle: -90, position: "insideLeft" }} />
          <Tooltip />
          <Legend />
          {boundaryLabel && (
            <ReferenceLine x={boundaryLabel} stroke="#94a3b8" strokeDasharray="4 4" label="Now" />
          )}
          <Line type="monotone" dataKey="actual" stroke="#2563eb" dot={false} name="Actual" connectNulls={false} />
          <Line type="monotone" dataKey="forecast" stroke="#f97316" dot={false} name="Forecast" connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
interface CostSimResult {
  load_name: string;
  baseline_start_hour: number;
  baseline_total_bill: number;
  recommended_start_hour: number | null;
  new_total_bill: number;
  estimated_savings: number;
  simulated_not_verified: boolean;
  no_savings_opportunity: boolean;
  note: string;
  tariff_source_note: string;
  baseline_period: { start: string | null; end: string | null; days_used: number };
  error?: string;
}

function CostSimulator({ building }: { building: string }) {
  const [powerKw, setPowerKw] = useState(5);
  const [durationHours, setDurationHours] = useState(2);
  const [baselineHour, setBaselineHour] = useState(10);
  const [result, setResult] = useState<CostSimResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function runSimulation() {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/cost-simulation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          building,
          power_kw: powerKw,
          duration_hours: durationHours,
          baseline_start_hour: baselineHour,
        }),
      });
      setResult(await res.json());
    } catch {
      setResult({ error: "Could not reach backend" } as CostSimResult);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-sm text-slate-500 mb-3">Cost simulator — {building}</p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <label className="text-sm text-slate-600">
          Appliance power (kW)
          <input
            type="number"
            value={powerKw}
            onChange={(e) => setPowerKw(Number(e.target.value))}
            className="mt-1 w-full rounded border border-slate-300 px-2 py-1"
          />
        </label>
        <label className="text-sm text-slate-600">
          Duration (hours)
          <input
            type="number"
            value={durationHours}
            onChange={(e) => setDurationHours(Number(e.target.value))}
            className="mt-1 w-full rounded border border-slate-300 px-2 py-1"
          />
        </label>
        <label className="text-sm text-slate-600">
          Current start hour (0-23)
          <input
            type="number"
            min={0}
            max={23}
            value={baselineHour}
            onChange={(e) => setBaselineHour(Number(e.target.value))}
            className="mt-1 w-full rounded border border-slate-300 px-2 py-1"
          />
        </label>
      </div>

      <button
        onClick={runSimulation}
        disabled={loading}
        className="mt-4 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? "Simulating..." : "Run simulation"}
      </button>

      {result?.error && <p className="mt-4 text-red-600">{result.error}</p>}

      {result && !result.error && (
        <div className="mt-4 rounded border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs uppercase tracking-wide text-slate-400">
            Simulated — not a verified or measured result
          </p>

          {result.no_savings_opportunity ? (
            <p className="mt-2 text-slate-700">
              <span className="font-semibold">No savings opportunity.</span> {result.note}
            </p>
          ) : (
            <p className="mt-2 text-slate-700">
              Moving this load to <span className="font-semibold">{result.recommended_start_hour}:00</span>{" "}
              could save an estimated{" "}
              <span className="font-semibold text-green-700">
                ₹{result.estimated_savings.toFixed(2)}
              </span>{" "}
              over this period.
            </p>
          )}

          <p className="mt-2 text-sm text-slate-500">
            Baseline bill: ₹{result.baseline_total_bill.toFixed(2)} — New bill: ₹
            {result.new_total_bill.toFixed(2)}
          </p>

          <p className="mt-3 text-xs text-slate-400">{result.tariff_source_note}</p>
        </div>
      )}
    </div>
  );
}
const HOUR_LABELS = Array.from({ length: 24 }, (_, i) => i);
const DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function UsageAnalysis({ building }: { building: string }) {
  const [heatmap, setHeatmap] = useState<number[][] | null>(null);
  const [maxVal, setMaxVal] = useState(1);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${API_URL}/history?building=${building}&days=90`);
        if (!res.ok) throw new Error("fetch failed");
        const data: { history: { timestamp: string; actual_kwh: number | null }[] } =
          await res.json();

        // sum[dow][hour] and count[dow][hour] to compute a mean per cell
        const sums = Array.from({ length: 7 }, () => Array(24).fill(0));
        const counts = Array.from({ length: 7 }, () => Array(24).fill(0));

        for (const point of data.history) {
          if (point.actual_kwh === null) continue;
          const d = new Date(point.timestamp);
          const jsDay = d.getDay(); // 0=Sun..6=Sat
          const dow = (jsDay + 6) % 7; // convert to 0=Mon..6=Sun
          const hour = d.getHours();
          sums[dow][hour] += point.actual_kwh;
          counts[dow][hour] += 1;
        }

        const means = sums.map((row, d) =>
          row.map((s, h) => (counts[d][h] > 0 ? s / counts[d][h] : 0))
        );
        setHeatmap(means);
        setMaxVal(Math.max(...means.flat(), 1));
      } catch {
        setError("Could not load usage data");
      }
    }
    load();
  }, [building]);

  if (error) return <p className="mt-6 text-red-600">{error}</p>;
  if (!heatmap) return <p className="mt-6 text-slate-500">Loading usage analysis...</p>;

  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-sm text-slate-500 mb-3">
        Usage pattern — {building} (avg kWh by hour and day of week, last 90 days)
      </p>
      <div className="overflow-x-auto">
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              <th className="p-1"></th>
              {HOUR_LABELS.map((h) => (
                <th key={h} className="p-1 text-slate-400 font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {DOW_LABELS.map((label, dow) => (
              <tr key={label}>
                <td className="p-1 pr-2 text-slate-500 font-medium">{label}</td>
                {heatmap[dow].map((val, hour) => {
                  const intensity = val / maxVal;
                  return (
                    <td
                      key={hour}
                      title={`${label} ${hour}:00 — ${val.toFixed(1)} kWh`}
                      style={{
                        backgroundColor: `rgba(37, 99, 235, ${intensity.toFixed(2)})`,
                      }}
                      className="w-6 h-6 border border-white"
                    />
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-slate-400">
        Darker = higher average demand for that hour/weekday combination.
      </p>
    </div>
  );
}
interface MetricsResponse {
  evaluation_method: string;
  production_model: string;
  results_by_building: Record<string, { naive_wape: number; rf_wape: number; gbm_wape: number }>;
  overall: { naive_wape: number; rf_wape: number; gbm_wape: number };
  note: string;
}

function ModelQuality() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/metrics`)
      .then((res) => res.json())
      .then(setMetrics)
      .catch(() => setError("Could not load metrics"));
  }, []);

  if (error) return <p className="mt-6 text-red-600">{error}</p>;
  if (!metrics) return <p className="mt-6 text-slate-500">Loading model quality...</p>;

  const buildings = Object.entries(metrics.results_by_building);

  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-sm text-slate-500">
        Model quality — {metrics.production_model}
      </p>
      <p className="text-xs text-slate-400 mb-3">{metrics.evaluation_method}</p>

      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-slate-500 border-b border-slate-200">
            <th className="py-2">Building</th>
            <th className="py-2 text-right">Naive WAPE</th>
            <th className="py-2 text-right">Random Forest</th>
            <th className="py-2 text-right font-semibold text-slate-700">GBM (production)</th>
          </tr>
        </thead>
        <tbody>
          {buildings.map(([name, m]) => (
            <tr key={name} className="border-b border-slate-100">
              <td className="py-2 capitalize">{name}</td>
              <td className="py-2 text-right text-slate-400">{m.naive_wape.toFixed(1)}%</td>
              <td className="py-2 text-right text-slate-500">{m.rf_wape.toFixed(1)}%</td>
              <td className="py-2 text-right font-semibold text-blue-700">{m.gbm_wape.toFixed(1)}%</td>
            </tr>
          ))}
          <tr className="font-semibold">
            <td className="py-2">All buildings</td>
            <td className="py-2 text-right text-slate-400">{metrics.overall.naive_wape.toFixed(1)}%</td>
            <td className="py-2 text-right text-slate-500">{metrics.overall.rf_wape.toFixed(1)}%</td>
            <td className="py-2 text-right text-blue-700">{metrics.overall.gbm_wape.toFixed(1)}%</td>
          </tr>
        </tbody>
      </table>

      <p className="mt-3 text-xs text-slate-400">{metrics.note}</p>
    </div>
  );
}
interface DataStatusResponse {
  building: string;
  data_range: { start: string; end: string };
  total_hours: number;
  missing_hours: number;
  missing_pct: number;
  longest_gap_hours: number;
  warning: string | null;
}

function DataStatus({ building }: { building: string }) {
  const [status, setStatus] = useState<DataStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/data-status?building=${building}`)
      .then((res) => res.json())
      .then(setStatus)
      .catch(() => setError("Could not load data status"));
  }, [building]);

  if (error) return <p className="mt-6 text-red-600">{error}</p>;
  if (!status) return <p className="mt-6 text-slate-500">Loading data status...</p>;

  return (
    <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-sm text-slate-500 mb-3">Data status — {building}</p>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <p className="text-xs text-slate-400">Data range</p>
          <p className="text-sm text-slate-700">
            {formatDate(status.data_range.start)} – {formatDate(status.data_range.end)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Total hours on record</p>
          <p className="text-sm text-slate-700">{status.total_hours.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Missing hours</p>
          <p className="text-sm text-slate-700">
            {status.missing_hours.toLocaleString()} ({status.missing_pct}%)
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Longest gap</p>
          <p className="text-sm text-slate-700">
            {status.longest_gap_hours.toLocaleString()} hours (~
            {Math.round(status.longest_gap_hours / 24)} days)
          </p>
        </div>
      </div>

      {status.warning && (
        <p className="mt-3 rounded bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-800">
          ⚠ {status.warning}
        </p>
      )}
    </div>
  );
}
function App() {
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [metaRes, forecastRes] = await Promise.all([
          fetch(`${API_URL}/metadata`),
          fetch(`${API_URL}/forecast`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ building: "academic", horizon_hours: 24 }),
          }),
        ]);

        if (!metaRes.ok || !forecastRes.ok) {
          throw new Error("Request failed");
        }

        setMetadata(await metaRes.json());
        setForecast(await forecastRes.json());
      } catch {
        setError("Backend not reachable — is uvicorn running?");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 p-8">
        <p className="text-slate-500">Loading...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-50 p-8">
        <h1 className="text-2xl font-semibold text-slate-800">EnergyWise</h1>
        <p className="mt-4 text-red-600">{error}</p>
      </div>
    );
  }

  const totalKwh = forecast!.forecast.reduce((sum, p) => sum + p.predicted_kwh, 0);
  const peak = forecast!.forecast.reduce((max, p) =>
    p.predicted_kwh > max.predicted_kwh ? p : max
  );

  const cards = [
    {
      label: "Next 24h forecast",
      value: `${totalKwh.toFixed(1)} kWh`,
    },
    {
      label: "Expected peak",
      value: `${peak.predicted_kwh.toFixed(1)} kWh`,
      sub: formatTime(peak.timestamp),
    },
    {
      label: "Model",
      value: metadata!.model_version.split(" - ")[0],
    },
    {
      label: "Training data through",
      value: formatDate(metadata!.training_range.end),
    },
  ];

  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <h1 className="text-2xl font-semibold text-slate-800">EnergyWise</h1>
      <p className="mt-1 text-slate-500">Academic building — demand forecast overview</p>

      <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card) => (
          <div key={card.label} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <p className="text-sm text-slate-500">{card.label}</p>
            <p className="mt-1 text-2xl font-semibold text-slate-800">{card.value}</p>
            {card.sub && <p className="text-sm text-slate-400">{card.sub}</p>}
          </div>
        ))}
      </div>
      <DemandChart building="academic" />
      <CostSimulator building="academic" />
      <UsageAnalysis building="academic" />
      <ModelQuality />
      <DataStatus building="academic" />
    </div>
  );
}

export default App;