/* Pan + zoom for every Mermaid <svg> */
document.addEventListener("DOMContentLoaded", () => {
  /* 1. Tell Mermaid to render after the page is parsed */
  mermaid.initialize({
    startOnLoad: true,
    securityLevel: "loose" // needed so svg-pan-zoom can inject buttons/handlers
  });

  /* 2. After Mermaid finishes, wrap every SVG */
  const activatePanZoom = () =>
    document.querySelectorAll(".mermaid svg").forEach(svg => {
      if (svg.dataset.panzoom) return;           // avoid double-init
      svgPanZoom(svg, {
        zoomEnabled: true,
        panEnabled: true,
        controlIconsEnabled: true,               // on-screen +/- buttons
        fit: true,
        center: true,
        minZoom: 0.25,
        maxZoom: 10
      });
      svg.dataset.panzoom = "true";
    });

  /* Mermaid emits a global event after each render */
  document.addEventListener("mermaidFinished", activatePanZoom);
});
