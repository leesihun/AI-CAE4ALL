/**
 * Axis ticks for the Studio's SVG line charts (Train Metrics, run comparison).
 *
 * Gridlines used to sit at five equal fractions of the data range padded by 8%,
 * so a loss curve was labelled 0.028497 / 0.018965 / -9.960e-5: values nobody
 * would pick, and a negative floor under a series that is never negative. The
 * fixed left margin then clipped that label's minus sign, and the chart printed
 * a plausible but wrong "9.960e-5". Ticks here are round (1, 2, 2.5 or 5 times a
 * power of ten), and the axis crosses zero only when the data does.
 */

const MULTIPLIERS = [1, 2, 2.5, 5, 10];

// 0.1 + 0.2 style residue would otherwise print as 0.30000000000000004.
const clean = value => {
  const rounded = Number(value.toPrecision(12));
  return rounded === 0 ? 0 : rounded;
};

/**
 * Round ticks covering [min, max]: {min, max, step, ticks}, where min and max
 * are the first and last tick, so the axis spans whole steps.
 *
 * `integer` keeps the step a whole number, for epoch or step counters.
 */
export function niceTicks(min, max, { target = 5, integer = false } = {}) {
  let low = Number(min);
  let high = Number(max);
  if (!Number.isFinite(low) || !Number.isFinite(high)) {
    return { min: 0, max: 1, step: .25, ticks: [0, .25, .5, .75, 1] };
  }
  if (low > high) [low, high] = [high, low];
  // A flat series still needs a span, and it stays on its own side of zero.
  // "Flat" includes a range of a few ulps: a constant LR smoothed by the EMA
  // comes back as 9.999999999999999e-5 .. 1e-4, whose step is so small that
  // low / step passes 2^53 and the tick loop below never advances.
  if (high - low <= 1e-6 * Math.max(Math.abs(low), Math.abs(high))) {
    const centre = low + (high - low) / 2;
    const half = integer ? 1 : Math.abs(centre) * .1 || .5;
    low = centre - half;
    high = centre + half;
  }
  let raw = (high - low) / Math.max(1, target - 1);
  if (integer) raw = Math.max(raw, 1);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = MULTIPLIERS
    .map(factor => clean(factor * magnitude))
    .find(candidate => candidate >= raw * (1 - 1e-9) && (!integer || Number.isInteger(candidate)));
  const first = Math.floor(low / step + 1e-9);
  const last = Math.ceil(high / step - 1e-9);
  const ticks = [];
  for (let index = first; index <= last; index += 1) ticks.push(clean(index * step));
  return { min: ticks[0], max: ticks[ticks.length - 1], step, ticks };
}

// The tolerance is relative to the step, not absolute: a 5e-7 step (a series
// around 1e-3 that moves by 2e-6) is not "within 1e-6 of the integer 0", and
// reading it that way labelled every tick "0".
function decimalsOf(step) {
  for (let decimals = 0; decimals < 15; decimals += 1) {
    const scaled = Math.abs(step) * 10 ** decimals;
    const rounded = Math.round(scaled);
    if (rounded !== 0 && Math.abs(scaled - rounded) < 1e-6 * scaled) return decimals;
  }
  return 15;
}

// 1.50e+4 -> 1.5e4. A trailing zero goes only when every tick can drop it, so
// the labels on one axis always share a precision.
function exponentialLabels(ticks, digits) {
  let precision = Math.max(0, digits - 1);
  const render = () => ticks.map(value => value.toExponential(precision).split("e"));
  let parts = render();
  while (precision > 0 && parts.every(([mantissa], index) => ticks[index] === 0 || mantissa.endsWith("0"))) {
    precision -= 1;
    parts = render();
  }
  return parts.map(([mantissa, exponent], index) => ticks[index] === 0 ? "0" : `${mantissa}e${exponent.replace("+", "")}`);
}

/**
 * Ticks for a log axis over strictly positive [min, max]: {min, max, ticks}.
 *
 * A loss that falls three decades is a straight line on a log axis and a spike
 * followed by a flat floor on a linear one, which is why the first epoch alone
 * used to decide how a whole run looked. Spans under three decades also get the
 * 2 and 5 marks, so a curve inside one decade still crosses a gridline; long
 * spans thin to at most seven decade ticks.
 */
export function logTicks(min, max) {
  let low = Math.min(Number(min), Number(max));
  let high = Math.max(Number(min), Number(max));
  if (!(low > 0) || !Number.isFinite(high)) return null;
  if (high <= low * (1 + 1e-6)) {
    // A flat series sits mid-axis, not on its floor. A range of a few ulps is
    // flat too: left alone, the trim below keeps a single tick, and a
    // one-tick axis has no span to place a point on.
    const centre = Math.sqrt(low) * Math.sqrt(high);
    low = centre / 2;
    high = centre * 2;
  }
  let first = Math.floor(Math.log10(low) + 1e-9);
  let last = Math.ceil(Math.log10(high) - 1e-9);
  if (last === first) last += 1;
  const decades = last - first;
  let ticks;
  if (decades < 3) {
    ticks = [];
    for (let exponent = first; exponent < last; exponent += 1) {
      for (const factor of [1, 2, 5]) ticks.push(clean(factor * 10 ** exponent));
    }
    ticks.push(clean(10 ** last));
    // Trim to the marks that just enclose the data.
    const start = Math.max(0, ticks.findLastIndex(value => value <= low * (1 + 1e-9)));
    const end = ticks.findIndex(value => value >= high * (1 - 1e-9));
    ticks = ticks.slice(start, end + 1);
  } else {
    const stride = Math.ceil(decades / 6);
    first = Math.floor(first / stride) * stride;
    last = Math.ceil(last / stride) * stride;
    ticks = [];
    for (let exponent = first; exponent <= last; exponent += stride) ticks.push(clean(10 ** exponent));
  }
  return { min: ticks[0], max: ticks[ticks.length - 1], ticks };
}

/** Labels for logTicks: plain decimals when every tick reads well that way. */
export function logTickLabels(ticks) {
  const plain = ticks.every(value => value >= 1e-3 && value < 1e7);
  return ticks.map(value => {
    if (plain || value === 1) return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
    const [mantissa, exponent] = value.toExponential(0).split("e");
    return `${mantissa}e${exponent.replace("+", "")}`;
  });
}

/** One label per tick, all at the precision the step needs and no more. */
export function tickLabels(ticks, step) {
  const largest = Math.max(...ticks.map(value => Math.abs(value)));
  if (largest >= 1e7 || (largest > 0 && largest < 1e-3)) {
    // A 2.5 step needs one digit past its own decade: 4.50e-5, 4.75e-5.
    const stepDigits = decimalsOf(clean(step / 10 ** Math.floor(Math.log10(step))));
    const digits = Math.min(8, Math.max(1, Math.floor(Math.log10(largest)) - Math.floor(Math.log10(step)) + 1 + stepDigits));
    return exponentialLabels(ticks, digits);
  }
  const decimals = decimalsOf(step);
  return ticks.map(value => value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }));
}
