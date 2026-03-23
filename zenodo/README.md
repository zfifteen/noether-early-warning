# Zenodo Metadata

This repository now contains two Zenodo-facing metadata artifacts:

- [`../.zenodo.json`](../.zenodo.json)  
  Use this as the GitHub-to-Zenodo metadata override for the software record.
- [`../technical-note/zenodo-deposit-metadata.json`](../technical-note/zenodo-deposit-metadata.json)  
  Use this as the API-ready metadata payload for the technical note deposit that will include [`../technical-note/export/noether-early-warning-technical-note.pdf`](../technical-note/export/noether-early-warning-technical-note.pdf).

## Recommended Deposit Order

1. Create the technical note draft deposit in Zenodo.
   Upload the PDF and apply [`../technical-note/zenodo-deposit-metadata.json`](../technical-note/zenodo-deposit-metadata.json).

2. Reserve or mint the technical note DOI.
   When the DOI is minted, add it to [`../.zenodo.json`](../.zenodo.json) as:

   ```json
   {
     "identifier": "REPLACE_WITH_TECHNICAL_NOTE_DOI",
     "scheme": "doi",
     "relation": "isDocumentedBy",
     "resource_type": "publication-technicalnote"
   }
   ```

3. Create the GitHub release `v0.1.0`.
   The recommended release body is prepared in [`v0.1.0_release_notes.md`](./v0.1.0_release_notes.md). Create the tag from the commit that freezes this DOI-prep state.

4. Create the software Zenodo record from the GitHub release.
   Once the software DOI is minted, add it back to the technical note metadata as:

   ```json
   {
     "identifier": "REPLACE_WITH_SOFTWARE_DOI",
     "scheme": "doi",
     "relation": "isSupplementedBy",
     "resource_type": "software"
   }
   ```

5. Update the published technical note record.
   Keep both the GitHub URL and the minted software DOI as supporting identifiers on the technical note record.

## Notes

- The technical note remains the publication object. The repository release remains the software object.
- The validated core claim package is `B1` through `B4`.
- The detector-latency sweep is part of the DOI-facing story as supporting robustness evidence, not as a fifth core benchmark.
- The recommended Zenodo record types are:
  - technical note: `upload_type=publication`, `publication_type=technicalnote`
  - repository release: `upload_type=software`
