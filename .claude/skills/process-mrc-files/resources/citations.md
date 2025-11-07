# MRC File Format Citations and References

This document tracks all external sources used in this skill for maintainability and version checking.

## Primary Specification

###  MRC2014 Standard Paper

- **Title:** "MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography"
- **Authors:** Cheng, A., Henderson, R., Mastronarde, D., Ludtke, S.J., Schoenmakers, R.H.M., Short, J., Marabini, R., Dallakyan, S., Agard, D., and Winn, M.
- **Journal:** Journal of Structural Biology
- **Volume/Issue:** 192(2)
- **Pages:** 146-150
- **Year:** 2015
- **DOI:** 10.1016/j.jsb.2015.04.002
- **PubMed ID:** PMID:25882513
- **PMC ID:** PMC4642651
- **URL:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4642651/
- **Last Checked:** 2025-11-07

## Official Format Specifications

### CCP-EM MRC Format Page

- **Organization:** Collaborative Computational Project for Electron cryo-Microscopy (CCP-EM)
- **Title:** "MRC2014 Image File Format"
- **URL:** https://www.ccpem.ac.uk/mrc-format/mrc2014/
- **General Format Page:** https://www.ccpem.ac.uk/mrc-format/
- **Maintainer:** CCP-EM consortium
- **Last Checked:** 2025-11-07

### IMOD MRC Format Documentation

- **Organization:** Boulder Laboratory for 3D Electron Microscopy of Cells
- **Title:** "MRC File Format"
- **URL:** https://bio3d.colorado.edu/imod/doc/mrc_format.txt
- **Software:** IMOD (Image Processing for Electron Microscopy)
- **Last Checked:** 2025-11-07

## Python Implementation

### mrcfile Library

- **Package Name:** mrcfile
- **Repository:** https://github.com/ccpem/mrcfile
- **Documentation:** https://mrcfile.readthedocs.io/
- **PyPI:** https://pypi.org/project/mrcfile/
- **Stable Version:** 1.5.4 (as of 2025-11-07)
- **Development Version:** 1.6.0b0
- **Maintainer:** CCP-EM (Colin Palmer, primary author)
- **License:** BSD 3-Clause
- **Python Support:** Python 3.x
- **Dependencies:** NumPy (primary), minimal others
- **Last Checked:** 2025-11-07

**Version Tracking:**
- Check PyPI for latest stable: https://pypi.org/project/mrcfile/#history
- Check GitHub releases: https://github.com/ccpem/mrcfile/releases

## C++ Implementation

### cisTEM MRC Implementation

- **Project:** cisTEMx
- **Repository:** https://github.com/StochasticAnalytics/cisTEMx
- **Primary Files:**
  - `src/core/mrc_header.h` - MRCHeader class
  - `src/core/mrc_header.cpp` - Implementation
  - `src/core/mrc_file.h` - MRCFile class
  - `src/core/mrc_file.cpp` - Implementation
- **Dependencies:**
  - ieee-754-half library (for FP16 support): `include/ieee-754-half/half.hpp`
- **Last Checked:** 2025-11-07

**Note:** This is internal to cisTEMx project; version tracking via git history.

## Related Software and Tools

### RELION (REgularised LIkelihood OptimisatioN)

- **Project:** RELION
- **Documentation:** https://relion.readthedocs.io/
- **Repository:** https://github.com/3dem/relion
- **Relevance:** Defines `.mrc` vs `.mrcs` convention for volumes vs stacks
- **Last Checked:** 2025-11-07

### EMAN2 (Electron Micrograph ANalysis)

- **Project:** EMAN2
- **Website:** https://blake.bcm.edu/emanwiki/EMAN2
- **Relevance:** Alternative EM processing suite with MRC support
- **Last Checked:** 2025-11-07

### Chimera / ChimeraX

- **Project:** UCSF Chimera and ChimeraX
- **Website:** https://www.cgl.ucsf.edu/chimerax/
- **Relevance:** Molecular visualization; origin field convention differences
- **Last Checked:** 2025-11-07

## Research Papers

### Cryo-EM Data Precision

- **Title:** "Addressing preferred specimen orientation in single-particle cryo-EM through tilting"
- **Relevance:** Discusses data type precision requirements in cryo-EM
- **PMC ID:** PMC9645247
- **URL:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9645247/
- **Last Checked:** 2025-11-07

## Historical Context

### Original MRC Format (pre-2000)

- **Origins:** Medical Research Council Laboratory of Molecular Biology (MRC-LMB), Cambridge, UK
- **Initial Use:** Electron density maps from X-ray crystallography
- **Evolution:** Adopted and extended by EM community in 1990s-2000s

### CCP4 Variant

- **Organization:** Collaborative Computational Project Number 4 (CCP4)
- **Website:** https://www.ccp4.ac.uk/
- **Relevance:** Crystallography-focused MRC variant with different conventions
- **Note:** MRC2014 attempted to unify EM and crystallography standards

## Version History Summary

| Version | Year | Key Changes |
|---------|------|-------------|
| Original MRC | Pre-2000 | Initial format, no endianness standard |
| MRC2000 | ~2000 | Added machine stamp field, standardization efforts |
| IMOD Variant | 2000s | EM-specific extensions, imodStamp/imodFlags |
| FEI Variant | 2000s-2010s | Microscope vendor-specific extended headers |
| MRC2014 | 2015 | Current standard: MODE 6 (uint16), EXTTYPE field, NVERSION clarification |

## Maintenance Notes

### When to Update This Document

1. **Annually** - Check all URLs are still valid
2. **When mrcfile releases new version** - Update version numbers
3. **When MRC format evolves** - Add new specification references
4. **When cisTEMx MRC code changes significantly** - Note API changes
5. **When interoperability issues discovered** - Add relevant tool documentation

### How to Check for Updates

**mrcfile:**
```bash
pip list | grep mrcfile  # Check installed version
pip install --upgrade mrcfile  # Upgrade if needed
```

**cisTEMx:**
```bash
cd /workspaces/cisTEMx
git log --oneline src/core/mrc_*.{h,cpp}  # Check recent changes
```

**Specifications:**
- Visit CCP-EM page: https://www.ccpem.ac.uk/mrc-format/
- Check for errata or updates to MRC2014 paper

## Contact Information

### For Format Questions

- **CCP-EM Discussion:** https://www.jiscmail.ac.uk/ccp4bb
- **mrcfile Issues:** https://github.com/ccpem/mrcfile/issues

### For cisTEMx Implementation Questions

- **Repository Issues:** https://github.com/StochasticAnalytics/cisTEMx/issues
- **Advisor:** Anwar (project PI)

## License Information

### This Skill

- **License:** Project-specific (cisTEMx)
- **Usage:** Internal to cisTEMx development

### Referenced Materials

- **MRC2014 Paper:** Academic publication (cite appropriately)
- **mrcfile Library:** BSD 3-Clause License
- **CCP-EM Specifications:** Public documentation (CCP-EM consortium)
- **IMOD Documentation:** Public documentation
- **cisTEMx Code:** Project license (check repository)

## Change Log for This Document

| Date | Change | Updated By |
|------|--------|------------|
| 2025-11-07 | Initial creation during MRC skill development | Claude (research synthesis) |

## Notes

- All URLs verified as of 2025-11-07
- Software versions current as of 2025-11-07
- This document should be reviewed and updated at least annually
- When citing in papers, use the MRC2014 paper (Cheng et al. 2015) as primary reference
