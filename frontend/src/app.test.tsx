import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { vi } from "vitest";

vi.mock("maplibre-gl", () => ({ default: {}, Map: class {}, Popup: class {} }));

import { Login } from "./App";
import { Answer } from "./pages/Command";
import { Async, ErrorBox, Provenance, Stat, Table } from "./components/ui";
import { colorExpression, heightExpression, plantsToFC, popupHtml } from "./components/PowerMap";
import { ApiError, api, getKey } from "./api";

afterEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

const mockFetch = (status: number, body: unknown) =>
  vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: status < 400, status, statusText: "x", json: async () => body } as Response);

test("api sends key header and parses structured errors", async () => {
  sessionStorage.setItem("poweriq_api_key", "k1");
  const f = mockFetch(403, { error: { code: "forbidden", message: "Requires role 'analyst' or higher.", request_id: "abc" } });
  await expect(api("/x")).rejects.toMatchObject({ status: 403, code: "forbidden", requestId: "abc" });
  expect((f.mock.calls[0][1] as RequestInit).headers).toMatchObject({ "X-API-Key": "k1" });
});

test("api reports network failure clearly", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fail"));
  await expect(api("/x")).rejects.toMatchObject({ code: "network_error" });
});

test("login rejects an invalid key and does not store it", async () => {
  mockFetch(401, { error: { code: "invalid_credentials", message: "Invalid API key." } });
  const done = vi.fn();
  render(<Login onDone={done} />);
  fireEvent.change(screen.getByLabelText("API key"), { target: { value: "bad" } });
  fireEvent.click(screen.getByRole("button", { name: /enter command centre/i }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Invalid API key.");
  expect(done).not.toHaveBeenCalled();
  expect(getKey()).toBe("");
});

test("login accepts a valid key", async () => {
  mockFetch(200, {});
  const done = vi.fn();
  render(<Login onDone={done} />);
  fireEvent.change(screen.getByLabelText("API key"), { target: { value: "demo-viewer-key" } });
  fireEvent.click(screen.getByRole("button", { name: /enter command centre/i }));
  await waitFor(() => expect(done).toHaveBeenCalled());
  expect(getKey()).toBe("demo-viewer-key");
});

test("Async shows loading, error and data states", () => {
  const { rerender } = render(<Async state={{ data: null, error: null, loading: true, reload: () => {} }}>{() => <div>data</div>}</Async>);
  expect(screen.getByRole("status")).toBeInTheDocument();
  rerender(<Async state={{ data: null, error: new ApiError(403, "forbidden", "nope"), loading: false, reload: () => {} }}>{() => <div>data</div>}</Async>);
  expect(screen.getByRole("alert")).toHaveTextContent("Not permitted");
  rerender(<Async state={{ data: { a: 1 }, error: null, loading: false, reload: () => {} }}>{() => <div>data</div>}</Async>);
  expect(screen.getByText("data")).toBeInTheDocument();
});

test("ErrorBox offers retry", () => {
  const retry = vi.fn();
  render(<ErrorBox error={new ApiError(0, "network_error", "down")} onRetry={retry} />);
  fireEvent.click(screen.getByText("Retry"));
  expect(retry).toHaveBeenCalled();
});

test("Table shows an informative empty state", () => {
  render(<Table cols={[{ key: "a", label: "A" }]} rows={[]} empty="Nothing flagged." />);
  expect(screen.getByText("Nothing flagged.")).toBeInTheDocument();
});

test("Stat and Provenance render labels", () => {
  render(<><Stat label="Demand now" value="1,291" unit="MW" /><Provenance origin="synthetic" model="v1" /></>);
  expect(screen.getByText("Demand now")).toBeInTheDocument();
  expect(screen.getByLabelText("data provenance")).toHaveTextContent("data: synthetic");
});

test("copilot answer separates facts, predictions, assumptions and recommendations", () => {
  render(<Answer a={{ mode: "local-template", answer: "Headline\n\nrest", measured_facts: ["F1"], model_predictions: ["P1"], assumptions: ["A1"], recommendations: ["R1"],
    citations: [{ citation: "[doc#0]", title: "Doc" }], timestamp: "t", model_version: "mv", uncertainty: "u", limitations: ["L1"], recommendation_records: [{ id: "REC-1" }], llm_status: "not used: x" }} />);
  for (const t of ["Measured facts", "Model predictions", "Assumptions", "advisory"]) expect(screen.getByText(new RegExp(t, "i"))).toBeInTheDocument();
  expect(screen.getByText(/\[doc#0\] Doc/)).toBeInTheDocument();
  expect(screen.getByText(/REC-1/)).toBeInTheDocument();
});

test("map expressions scale with metric and use a diverging scale for net flow", () => {
  const c = colorExpression("demand_mw", 200);
  expect(c[0]).toBe("interpolate");
  expect(JSON.stringify(c)).toContain("demand_mw");
  const d = colorExpression("net_flow_mw", 100);
  expect(d[3]).toBe(-100);
  const h = heightExpression("demand_mw", 200, 170000, true);
  expect(JSON.stringify(h)).toContain("sqrt");
  expect(JSON.stringify(heightExpression("demand_mw", 200, 170000, false))).not.toContain("sqrt");
});

test("plant and popup helpers", () => {
  const fc = plantsToFC([{ plant_id: "X", source: "wind", dispatch_mw: 100, lon: 36, lat: 1 }]);
  expect(fc.features[0].geometry.coordinates[0]).toHaveLength(5);
  expect(fc.features[0].properties.color).toBeTruthy();
  const html = popupHtml({ cell_id: "C001", region: "NORTH_RIFT", bus_id: "LESSOS", demand_mw: 12.34 }, { hour: 0, time: "", metrics: { demand_mw: "Demand (MW)" }, max: {} }, "demand_mw");
  expect(html).toContain("C001"); expect(html).toContain("12.3");
});
