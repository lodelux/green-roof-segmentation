# Green Roof Segmentation

This is a standalone portfolio presentation of the semantic-segmentation component from a two-person university project completed in 2025.

## Scope and attribution

- Keep this repository limited to the geospatial imagery and segmentation pipeline. The separate optimization component from the original team project is intentionally out of scope.
- Preserve the two-person university-project context and the link to the original team repository.
- Do not imply that the project detected existing green roofs. The municipal labels identify roofs considered potentially suitable for greening.
- Do not claim that raster-to-vector polygon conversion was implemented. It is documented only as a next step.
- Do not add a license without confirming the rights and agreement of both original contributors.

## Engineering conventions

- Never commit API keys, generated datasets, model weights, virtual environments, or `node_modules`.
- Do not commit provider-owned aerial imagery. Full datasets should be regenerated locally from the public vector source and a properly licensed imagery account.
- Use environment-local configuration for Azure Maps and retain `web/config.example.js` as the public template.
- Prefer deterministic preprocessing, explicit validation, and geographically separated evaluation when extending the experiments.

## Verification

- Run `uv run --extra dev pytest` for the Python data-processing tests.
- Run `python -m compileall src scripts` for Python syntax checks.
- Run `npm ci` and `npm run check` for the JavaScript capture script.
- Scan the complete Git history for credentials before publishing.
