/**
 * Named aliases over the GENERATED schema (./schema.d.ts). Nothing here is
 * hand-written shape: every type is a lookup into what openapi-typescript
 * produced from pleroma/serve/api-schema.json, so a contract change surfaces
 * as a type error, not as a silent `undefined` at runtime.
 */
import type { components, operations } from "./schema";

type Schemas = components["schemas"];

export type InfoResponse = Schemas["InfoResponse"];
export type ApiInfo = Schemas["ApiInfo"];
export type StateResponse = Schemas["StateResponse"];
export type SessionsResponse = Schemas["SessionsResponse"];
export type SessionRow = Schemas["SessionRow"];
export type ProgressResponse = Schemas["ProgressResponse"];
export type LoomProgress = Schemas["LoomProgress"];
export type ChatBody = Schemas["ChatBody"];
export type ChatResponse = Schemas["ChatResponse"];
export type LoomBody = Schemas["LoomBody"];
export type LoomResponse = Schemas["LoomResponse"];
export type Future = Schemas["Future"];
export type FutureScores = Schemas["FutureScores"];
export type WearBody = Schemas["WearBody"];
export type WearResponse = Schemas["WearResponse"];
export type UnwearResponse = Schemas["UnwearResponse"];
export type RestoreResponse = Schemas["RestoreResponse"];
export type WornPublic = Schemas["WornPublic"];
export type HistoryMessage = Schemas["HistoryMessage"];
export type DoseBandInfo = Schemas["DoseBandInfo"];
export type EffectiveAlpha = Schemas["EffectiveAlpha"];
export type Spread = Schemas["Spread"];
export type ErrorBody = Schemas["ErrorBody"];
export type ErrorCode = Schemas["ErrorCode"];

/** The two conversation branches. `loom` bends under a wear; `base` never does. */
export type Branch = NonNullable<ChatBody["branch"]>;
export const BRANCHES: readonly Branch[] = ["loom", "base"] as const;

/** The 200 JSON body of an operation. */
export type OkJson<Op> = Op extends { responses: { 200: { content: { "application/json": infer R } } } }
  ? R
  : never;
/** The JSON request body of an operation. */
export type JsonBody<Op> = Op extends { requestBody: { content: { "application/json": infer B } } }
  ? B
  : never;

export type Ops = operations;
