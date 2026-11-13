import type { GlobePoint, GlobeTrack } from "./Globe";

export const PATH_ORIGINAL = "#8a93ad";
export const PATH_MOVED = "#31d6a8";
export const PATH_BURN = "#ffc247";
export const PATH_TCA = "#ff5468";

/** A few km is far below a pixel on the globe, so the manoeuvred track is
 *  drawn with its offset from the original track exaggerated by this much
 *  (the legend says so; tables show the real distances). */
export const PATH_EXAGGERATION = 100;

function exaggerate(orig: [number, number, number], moved: [number, number, number]): [number, number, number] {
  let dlon = moved[1] - orig[1];
  if (dlon > 180) dlon -= 360;
  if (dlon < -180) dlon += 360;
  const lon = orig[1] + PATH_EXAGGERATION * dlon;
  return [orig[0] + PATH_EXAGGERATION * (moved[0] - orig[0]), ((lon + 540) % 360) - 180,
          orig[2] + PATH_EXAGGERATION * (moved[2] - orig[2])];
}

/** Globe layers for avoidance paths: the original track (faint), the
 *  manoeuvred one (bright), and markers at the burn and at closest approach. */
export function pathLayers(paths: any[]): { tracks: GlobeTrack[]; markers: GlobePoint[] } {
  const tracks: GlobeTrack[] = [];
  const markers: GlobePoint[] = [];
  for (const p of paths ?? []) {
    tracks.push({ points: p.original, color: PATH_ORIGINAL, opacity: 0.55 });
    const moved = p.maneuvered.map((m: [number, number, number], i: number) => exaggerate(p.original[i] ?? m, m));
    tracks.push({ points: moved, color: PATH_MOVED, opacity: 0.95 });
    const [bl, bo, ba] = p.burn_point;
    const [tl, to, ta] = exaggerate(p.tca_point_original, p.tca_point);
    markers.push({ lat: bl, lon: bo, alt_km: ba, color: PATH_BURN, highlight: true });
    markers.push({ lat: tl, lon: to, alt_km: ta, color: PATH_TCA, highlight: true });
  }
  return { tracks, markers };
}

export const pathLegend = (paths: any[]) => (paths?.length ? [
  { color: PATH_ORIGINAL, label: "track without the burn" },
  { color: PATH_MOVED, label: `track with the burn (burn, TCA, return; offset x${PATH_EXAGGERATION} to be visible)` },
  { color: PATH_BURN, label: "burn" },
  { color: PATH_TCA, label: "closest approach" },
] : []);
