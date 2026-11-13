import type { GlobePoint, GlobeTrack } from "./Globe";

export const PATH_ORIGINAL = "#8a93ad";
export const PATH_MOVED = "#31d6a8";
export const PATH_BURN = "#ffc247";
export const PATH_TCA = "#ff5468";
export const PATH_NOW = "#ffffff";

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

/** Globe layers for avoidance paths. The satellite's own dot is drawn at
 *  its true position, which lies on the grey (unexaggerated) track; the
 *  green track is the burn's effect with its offset exaggerated, as a hint.
 *  Both tracks are split at "now": the part already flown is dimmed, the part
 *  ahead is bright, and a white marker sits at "now" on each. */
export function pathLayers(paths: any[]): { tracks: GlobeTrack[]; markers: GlobePoint[] } {
  const tracks: GlobeTrack[] = [];
  const markers: GlobePoint[] = [];
  for (const p of paths ?? []) {
    const moved: [number, number, number][] = p.maneuvered.map((m: [number, number, number], i: number) => exaggerate(p.original[i] ?? m, m));
    const k: number = p.now_index ?? -1;
    for (const [pts, color, dim, bright] of [[p.original, PATH_ORIGINAL, 0.2, 0.6], [moved, PATH_MOVED, 0.25, 0.95]] as const) {
      if (k >= 0) tracks.push({ points: pts.slice(0, k + 1), color, opacity: dim });
      tracks.push({ points: pts.slice(Math.max(k, 0)), color, opacity: bright });
    }
    const [bl, bo, ba] = p.burn_point;
    const [tl, to, ta] = exaggerate(p.tca_point_original, p.tca_point);
    markers.push({ lat: bl, lon: bo, alt_km: ba, color: PATH_BURN, highlight: true });
    markers.push({ lat: tl, lon: to, alt_km: ta, color: PATH_TCA, highlight: true });
    if (k >= 0 && k < moved.length) {
      const [nl, no, na] = moved[k];
      markers.push({ lat: nl, lon: no, alt_km: na, color: PATH_NOW, highlight: true });
      const [ol, oo, oa] = p.original[k];
      markers.push({ lat: ol, lon: oo, alt_km: oa, color: PATH_NOW, highlight: true });
    }
  }
  return { tracks, markers };
}

export const pathLegend = (paths: any[]) => (paths?.length ? [
  { color: PATH_ORIGINAL, label: "true track (the satellite's dot rides this one)" },
  { color: PATH_MOVED, label: `effect of the burn, offset x${PATH_EXAGGERATION} to be visible` },
  { color: PATH_NOW, label: "now (dimmed: already flown)" },
  { color: PATH_BURN, label: "burn" },
  { color: PATH_TCA, label: "closest approach" },
] : []);
