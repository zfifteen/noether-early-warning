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
   Completed. The technical note DOI is:

   - [10.5281/zenodo.19184906](https://doi.org/10.5281/zenodo.19184906)

   [`.zenodo.json`](../.zenodo.json) now includes:

   ```json
   {
     "identifier": "10.5281/zenodo.19184906",
     "scheme": "doi",
     "relation": "isDocumentedBy",
     "resource_type": "publication-technicalnote"
   }
   ```

3. Create the GitHub release `v0.1.0`.
   Completed. The release body used for the archived release is preserved in [`v0.1.0_release_notes.md`](./v0.1.0_release_notes.md).

4. Create the software Zenodo record from the GitHub release.
   Completed. The software DOI is:

   - [10.5281/zenodo.19184861](https://doi.org/10.5281/zenodo.19184861)

   [`../technical-note/zenodo-deposit-metadata.json`](../technical-note/zenodo-deposit-metadata.json) now includes:

   ```json
   {
     "identifier": "10.5281/zenodo.19184861",
     "scheme": "doi",
     "relation": "isSupplementedBy",
     "resource_type": "software"
   }
   ```

5. Update the published technical note record.
   Completed. Keep both the GitHub URL and the minted software DOI as supporting identifiers on the technical note record.

## Notes

- The technical note remains the publication object. The repository release remains the software object.
- The validated core claim package is `B1` through `B4`.
- The detector-latency sweep is part of the DOI-facing story as supporting robustness evidence, not as a fifth core benchmark.
- The DOI pair is now fully established:
  - technical note DOI: [10.5281/zenodo.19184906](https://doi.org/10.5281/zenodo.19184906)
  - software DOI: [10.5281/zenodo.19184861](https://doi.org/10.5281/zenodo.19184861)
- The recommended Zenodo record types are:
  - technical note: `upload_type=publication`, `publication_type=technicalnote`
  - repository release: `upload_type=software`
