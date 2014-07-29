rdkit-utils [![Build Status](https://travis-ci.org/skearnes/rdkit-utils.svg?branch=master)](https://travis-ci.org/skearnes/rdkit-utils)
===========

Cheminformatics utilities based on the [RDKit](http://www.rdkit.org/)

Highlights
----------

### High-level molecule reading and writing
Read and write multiple molecule file formats using the same interface. The `MolReader` class automatically __perceives conformers__ and can optionally __remove salts__.

```python
from rdkit_utils.serial import MolReader

# read a gzipped SDF file
reader = MolReader()
with reader.open('molecules.sdf.gz') as mols:
    for mol in mols:
        ...

# read SMILES
reader = MolReader()
with reader.open('molecules.smi') as mols:
    for mol in mols:
        ...
        
# read from a file-like object
with open('molecules.sdf') as f:
    for mol in MolReader(f, mol_format='sdf'):
        ...
```

### Conformer generation
Generate conformers with minimization _prior_ to pruning. The `ConformerGenerator` class starts with a pool of conformers and prunes out conformers within an RMSD threshold.

```python
from rdkit_utils import conformers, serial

reader = serial.MolReader()
mols = reader.open('molecules.sdf')

engine = conformers.ConformerGenerator(max_conformers=10)
expanded = []
for mol in mols:
    expanded.append(engine.generate_conformers(mol))
    ...
```
