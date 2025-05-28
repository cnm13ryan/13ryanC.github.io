# MyResearchNotes — AI research blog powered by Hugo

![Deploy](https://github.com/13ryanC/13ryanC.github.io/actions/workflows/gh-pages.yml/badge.svg?style=flat-square)

<details>
<summary>Table of Contents</summary>

- [Overview](#overview--why)
- [Quickstart](#quickstart)
- [Documentation & Resources](#documentation--resources)
- [Configuration / Usage Deep-Dive](#configuration--usage-deep-dive)
- [Project Roadmap & Status](#project-roadmap--status)
- [Contributing](#contributing)
- [Community & Support](#community--support)
- [Security](#security)
- [License](#license)
- [Acknowledgements / Sponsors](#acknowledgements--sponsors)

</details>

## Overview / Why?

MyResearchNotes publishes research articles and tutorials on artificial intelligence. The site uses the
[Hugo](https://gohugo.io/) static site generator with the feature-rich
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme. Key features include:

- KaTeX support for math notation ✅
- Mermaid diagrams for architecture visuals ✅
- GitHub Actions workflow for automatic deployment ✅

## Quickstart

```bash
# install Hugo (extended)
git clone https://github.com/13ryanC/13ryanC.github.io.git
cd 13ryanC.github.io
git submodule update --init --recursive
hugo server -D
```
Visit <http://localhost:1313> to view the site.

## Documentation & Resources

- [Product requirements](blog_platform_prd.md)
- [System design](MyResearchNotes_system_design.md)
- Diagrams: `MyResearchNotes_class_diagram.mermaid`, `MyResearchNotes_sequence_diagram.mermaid`

## Configuration / Usage Deep-Dive

Configuration lives in [config.toml](config.toml). Common settings:

```toml
title = "My Research Notes"
baseURL = "https://13ryanC.github.io/"
[params]
  math = true
  ShowCodeCopyButtons = true
```

Update `baseURL` for your own GitHub Pages URL and tweak theme parameters as needed.

## Project Roadmap & Status

![status](https://img.shields.io/badge/status-beta-blue.svg?style=flat-square)

Future plans include adding a commenting system and analytics.

## Contributing

Contributions are welcome! Please open an issue or pull request.
A `CONTRIBUTING.md` file is not yet available — **TODO add guidelines.**

## Community & Support

Use GitHub Issues for questions or feedback.

## Security

Please report security concerns via GitHub Issues.
A dedicated `SECURITY.md` is missing — **TODO add policy.**

## License

**TODO: Add a license file.** Until then this repository should be treated as
all rights reserved.

## Acknowledgements / Sponsors

Built with the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme and inspired by the wider open-source community.

