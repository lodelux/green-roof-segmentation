# Green Roof Segmentation

This is the semantic-segmentation project developed by Lorenzo De Luca and Edoardo Guida as a two-person university project in 2025.

## Scope and attribution

- Keep this repository limited to the geospatial imagery and segmentation pipeline.
- Preserve the two-person university-project context.
- Do not imply that the project detected existing green roofs. The municipal labels identify roofs considered potentially suitable for greening.
- Do not claim that raster-to-vector polygon conversion was implemented. It is documented only as a next step.
- Do not add a license without confirming the rights and agreement of both original contributors.

## Engineering conventions

- Never commit API keys, generated datasets, model weights, virtual environments, or `node_modules`.
- Keep only the small set of historical figures used by the original project report. Do not commit the generated training dataset or model weights.
- Use environment-local configuration for Azure Maps and retain `web/config.example.js` as the public template.
- Prefer deterministic preprocessing, explicit validation, and geographically separated evaluation when extending the experiments.

## Verification

- Run `uv run --extra dev pytest` for the Python data-processing tests.
- Run `python -m compileall src scripts` for Python syntax checks.
- Run `npm ci` and `npm run check` for the JavaScript capture script.
- Scan the complete Git history for credentials before publishing.
