/* assets/js/mermaid-hydrate.js */
function promoteMermaidCodeBlocks() {
  document
    .querySelectorAll('pre > code.language-mermaid')
    .forEach(code => {
      const pre   = code.parentElement;
      const div   = document.createElement('div');
      div.className   = 'mermaid';
      div.textContent = code.textContent;   // keep the raw diagram text
      pre.replaceWith(div);
    });
}
