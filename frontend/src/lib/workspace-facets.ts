import type { Document } from "@/lib/api";

/**
 * What a workspace actually contains, so the UI never prompts the user with a
 * company or fiscal year that has not been uploaded.
 */
export interface Facets {
  /** Company names, the one with the newest filing first. */
  companies: string[];
  /** Fiscal years across all named documents, descending (form suggestions). */
  years: number[];
  /** Fiscal years per company, descending. */
  yearsByCompany: Record<string, number[]>;
  /**
   * Live documents in the workspace, whether or not they name a company. A
   * corpus uploaded without metadata has facets of its own and none; only the
   * second one means "upload something first".
   */
  nDocuments: number;
}

/** Only searchable documents should drive the prompts. */
function isLive(doc: Document): boolean {
  return doc.status === "active";
}

export function facetsFromDocuments(docs: Document[]): Facets {
  const yearsByCompany: Record<string, number[]> = {};
  for (const doc of docs) {
    if (!isLive(doc)) continue;
    const name = doc.company.trim();
    if (!name) continue;
    const years = (yearsByCompany[name] ??= []);
    if (typeof doc.year === "number" && !years.includes(doc.year)) years.push(doc.year);
  }
  for (const years of Object.values(yearsByCompany)) years.sort((a, b) => b - a);

  const companies = Object.entries(yearsByCompany)
    .sort((a, b) => (b[1][0] ?? -1) - (a[1][0] ?? -1) || a[0].localeCompare(b[0]))
    .map(([name]) => name);

  const years = [...new Set(Object.values(yearsByCompany).flat())].sort((a, b) => b - a);

  return {
    companies,
    years,
    yearsByCompany,
    nDocuments: docs.filter(isLive).length,
  };
}

export function chatSuggestions(f: Facets): string[] {
  const [company, other] = f.companies;
  if (!company) return [];
  const [latest, previous] = f.yearsByCompany[company] ?? [];

  const out: string[] = [];
  out.push(
    latest
      ? `What was ${company}'s net income in fiscal year ${latest}?`
      : `What was ${company}'s latest net income?`,
  );
  out.push(`What are the main risks ${company} faces?`);
  if (latest && previous) {
    out.push(`Compare ${company}'s gross margin between ${previous} and ${latest}.`);
  } else if (other) {
    out.push(`Compare ${company} and ${other} on revenue growth.`);
  } else {
    out.push(
      latest
        ? `Summarise ${company}'s ${latest} results.`
        : `Summarise the newest results for ${company}.`,
    );
  }
  return out.slice(0, 4);
}
