# Format przekazania A↔B

Każdy marker gotowości to JSON UTF-8. Nie tworzyć pustych plików READY. Minimalny envelope:

```json
{
  "schema_version": 1,
  "study_id": "data-ood-20261030",
  "generation": 1,
  "owner": "A",
  "phase": "A_UNIVERSE_FROZEN",
  "status": "complete",
  "contract_sha256": "sha256 pliku COORDINATION.md",
  "artifacts": [
    {"role": "training_universe", "path": "C:/.../universe.jsonl", "sha256": "...", "bytes": 123},
    {"role": "protected_index", "path": "C:/.../protected.jsonl", "sha256": "...", "bytes": 456}
  ]
}
```

Ścieżka `path` jest absolutną ścieżką Windows z `/`. WSL normalizuje ją `local_path`. Wewnątrz zamrożonych protokołów galerie mogą zachować `/mnt/c/...` zgodnie z dotychczasowym runnerem WSL. Przed użyciem sprawdzać tożsamość pliku i SHA. Teksty w przykładzie są placeholderami schematu, nie markerem gotowości.

Wymagane role:

| Marker | Role i dodatkowe pola |
|---|---|
| COMMON_READY | `common_helper`, `helper_usage`, `helper_verification`; API i jego wersja |
| A_INPUTS_FROZEN | `VN_export`, `Wiki_export`, `legacy_train_inputs`; faktyczne liczniki i polityka wyboru eksportu |
| A_UNIVERSE_FROZEN | `training_universe`, `protected_index`, `image_audit`, `study_registration`; fingerprinty obejmują wszystkie dopuszczone warianty |
| B_EVAL_FROZEN | `evaluation_registration`, po jednym `manifest:<id>` i `protocol:<id>` na dataset; zależność `A_UNIVERSE_FROZEN` wraz z SHA markera |
| A_FINALISTS_FROZEN | `training_report`, `selection_decision`; dodatkowe `models` i `representative_model_ids` |
| A_GPU_DONE | zależność od SHA `A_FINALISTS_FROZEN`; `further_A_cuda_planned=false` |
| B_RESULTS_READY | raporty, wyniki i ich SHA; zależności od markerów finalistów A i ewaluacji B |

`models` w markerze A ma dla każdego checkpointu: `id`, `variant`, `seed`, `path`, `sha256`, `training_manifest_sha256`, `trained_on_Wiki`, `representative`, `encoder_revisions`, `text_prefix`. Pole `representative_model_ids` wskazuje dokładnie jednego reprezentanta każdego R0/R1/wybranyWiki; wszystkie dziewięć checkpointów jest zamkniętych przed score OOD. `trained_on_Wiki` służy oznaczeniu ID/OOD per model, nie zmianie starego baseline'u.

Wiersz `training_universe` ma co najmniej: kanoniczny identyfikator obrazu, zweryfikowaną ścieżkę, SHA pliku/pikseli, pHash, tożsamości źródeł/rodziców, znormalizowane teksty i referencje do dowodów. Nie ujawnia wyników nowych testów B. `protected_index` odróżnia istniejący train, dev/val/eval i blacklisty; B odrzuca sumę tych grup, nie tylko finalnie wylosowane obrazy A.

Sesje nie zmieniają generacji w połowie trwającego porównania. Zmiana zależnego pliku wymaga nowej wersji markera i jawnego ponownego zamknięcia downstream, przed jakimkolwiek wynikiem. Własne `A_STATUS.json`/`B_STATUS.json` mogą być mutowalne i wskazywać `waiting_for`, ale nie zastępują gotowych markerów.
