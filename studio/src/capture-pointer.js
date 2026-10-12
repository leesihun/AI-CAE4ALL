// Optional pointer feedback for recordings of the actual Studio page.
// It follows real browser input; it never dispatches input or changes a result.
if (new URLSearchParams(location.search).get("capture") === "1") {
  const pointer = document.createElement("div");
  pointer.setAttribute("aria-hidden", "true");
  pointer.style.cssText = "position:fixed;z-index:2147483647;pointer-events:none;left:0;top:0;display:none;width:24px;height:30px;filter:drop-shadow(0 1px 2px #0008)";
  pointer.innerHTML = '<svg width="24" height="30" viewBox="0 0 24 30"><path d="M3 2 L3 23 L9 18 L14 28 L18 26 L13 16 L22 15 Z" fill="white" stroke="#10251c" stroke-width="1.5"/></svg>';
  document.body.append(pointer);
  const move = event => {
    pointer.style.display = "block";
    pointer.style.transform = `translate(${event.clientX}px,${event.clientY}px)`;
  };
  document.addEventListener("pointermove", move, { passive: true });
  document.addEventListener("pointerdown", event => {
    move(event);
    const ring = document.createElement("div");
    ring.style.cssText = `position:fixed;z-index:2147483646;pointer-events:none;left:${event.clientX - 17}px;top:${event.clientY - 17}px;width:34px;height:34px;border:3px solid #acd956;border-radius:50%;box-sizing:border-box;box-shadow:0 0 0 4px #10251c22`;
    document.body.append(ring);
    ring.animate([{ opacity: .95, transform: "scale(.7)" }, { opacity: 0, transform: "scale(1.6)" }], { duration: 550 }).finished.then(() => ring.remove());
  }, { passive: true });
}
