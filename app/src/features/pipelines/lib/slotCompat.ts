import type { PluginSlot } from "@/lib/api";

/**
 * Returns true when an output slot's type can feed an input slot's type.
 * Currently: exact match only. "None" slots are never compatible with anything
 * except themselves (source/sink boundary guards).
 */
export function areSlotsCompatible(
  outputSlotType: PluginSlot["type"],
  inputSlotType: PluginSlot["type"],
): boolean {
  return outputSlotType === inputSlotType;
}

const SLOT_COLORS: Record<PluginSlot["type"], string> = {
  None: "#c9c5be",
  RawText: "#26887d",
  RawTextStream: "#00d579",
  Entities: "#002147",
  LinkedEntities: "#eeff41",
};

/** Returns a hex color for the given slot type, drawn from the theme palette. */
export function slotColor(slotType: PluginSlot["type"]): string {
  return SLOT_COLORS[slotType] ?? "#c9c5be";
}
