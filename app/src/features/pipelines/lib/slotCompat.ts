import type { PluginSlot } from "@/lib/api";

/**
 * Slot subtype relationships (mirrors the backend `slot_accepts`): a producer
 * type on the left may feed a consumer declaring the type on the right, even
 * without an exact match. `Entities` → `LinkedEntities` holds because a
 * LinkedEntity is an Entity with optional matches, so a pipeline can end on
 * NER output (text → ner → results) with no NEL stage.
 */
const SLOT_SUBTYPES: Partial<Record<PluginSlot["type"], PluginSlot["type"][]>> = {
  Entities: ["LinkedEntities"],
};

/**
 * Returns true when an output slot's type can feed an input slot's type —
 * exact match, or a declared subtype relationship. "None" slots are never
 * compatible with anything except themselves (source/sink boundary guards).
 */
export function areSlotsCompatible(
  outputSlotType: PluginSlot["type"],
  inputSlotType: PluginSlot["type"],
): boolean {
  if (outputSlotType === inputSlotType) {
    return true;
  }
  return (SLOT_SUBTYPES[outputSlotType] ?? []).includes(inputSlotType);
}

const SLOT_COLORS: Record<PluginSlot["type"], string> = {
  None: "#c9c5be",
  RawText: "#26887d",
  RawTextStream: "#00d579",
  Entities: "#002147",
  LinkedEntities: "#e67e22",
};

/** Returns a hex color for the given slot type, drawn from the theme palette. */
export function slotColor(slotType: PluginSlot["type"]): string {
  return SLOT_COLORS[slotType] ?? "#c9c5be";
}
