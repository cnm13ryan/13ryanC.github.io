// Enable pan+zoom on every Mermaid <svg>
(function () {
  const panZoomOptions = {
    zoomEnabled: true,
    panEnabled:  true,
    controlIconsEnabled: true,
    fit: true,
    center: true,
    minZoom: 0.25,
    maxZoom: 10
  };

  // 1️⃣  Run once now (in case Mermaid already finished)
  enableOnAll();

  // 2️⃣  Run again every time a new diagram is rendered
  document.addEventListener("mermaidAfterRender", enableOnAll);

  function enableOnAll() {
    document.querySelectorAll(".mermaid svg").forEach(svg => {
      if (svg.dataset.panzoom) return;      // avoid double-init
      svgPanZoom(svg, panZoomOptions);
      svg.dataset.panzoom = "yes";
    });
  }
})();
