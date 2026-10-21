/**
 * The project has no browser test runner, so the workspace-facet logic that
 * drives the chat prompts and the report form is checked here instead: compile
 * the real module with the repo's own TypeScript, then assert on its output.
 *
 *   npm run check:facets
 */
const assert = require("assert");
const path = require("path");

const { facetsFromDocuments, chatSuggestions } = require(
  path.join(__dirname, "..", "node_modules", ".facets-check", "workspace-facets.js"),
);

const doc = (o) => ({
  id: o.id, workspace_id: "ws", title: o.title || "t", filename: "f.pdf",
  doc_type: "annual_report", company: o.company ?? "", ticker: "",
  year: o.year ?? null, quarter: "", section: "", status: o.status || "active",
  parse_meta: {}, created_at: "2026-01-01T00:00:00Z",
});

assert.deepStrictEqual(facetsFromDocuments([]), {
  companies: [], years: [], yearsByCompany: {}, nDocuments: 0,
});
assert.deepStrictEqual(chatSuggestions(facetsFromDocuments([])), []);

const f = facetsFromDocuments([
  doc({ id: "1", company: "Old Co", year: 2019, status: "archived" }),
  doc({ id: "2", company: "Aramco", year: 2024 }),
  doc({ id: "3", company: "Aramco", year: 2023 }),
  doc({ id: "4", company: "  SABIC  ", year: 2021 }),
  doc({ id: "5", company: "", year: 2020 }),
]);
assert.deepStrictEqual(f, {
  companies: ["Aramco", "SABIC"],
  years: [2024, 2023, 2021],
  yearsByCompany: { Aramco: [2024, 2023], SABIC: [2021] },
  nDocuments: 4,   // doc 1 is archived, doc 5 is live but unnamed
});

// A corpus uploaded without any company metadata still has documents: the form
// must not tell the user to upload something. This is the PIF shape.
assert.deepStrictEqual(facetsFromDocuments([
  doc({ id: "1", year: 2024 }), doc({ id: "2", year: 2023 }),
]), { companies: [], years: [], yearsByCompany: {}, nDocuments: 2 });

assert.deepStrictEqual(chatSuggestions(f), [
  "What was Aramco's net income in fiscal year 2024?",
  "What are the main risks Aramco faces?",
  "Compare Aramco's gross margin between 2023 and 2024.",
]);

// One filing for the company -> no invented comparison year, even though the
// workspace holds another company's filing.
assert.deepStrictEqual(chatSuggestions(facetsFromDocuments([
  doc({ id: "1", company: "Aramco", year: 2024 }),
])), [
  "What was Aramco's net income in fiscal year 2024?",
  "What are the main risks Aramco faces?",
  "Summarise Aramco's 2024 results.",
]);

assert.match(chatSuggestions(facetsFromDocuments([
  doc({ id: "1", company: "Aramco", year: 2024 }),
  doc({ id: "2", company: "SABIC", year: 2023 }),
]))[2], /Compare Aramco and SABIC on revenue growth/);

// No fiscal-year metadata at all -> prompts stay answerable.
const noYear = chatSuggestions(facetsFromDocuments([doc({ id: "1", company: "Aramco" })]));
assert.deepStrictEqual(noYear[0], "What was Aramco's latest net income?");
assert.match(noYear[2], /Summarise the newest results for Aramco/);

console.log("workspace facets + chat suggestions: ok");
