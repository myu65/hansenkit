"""Oriented chemical assembly and bounded chain features, separate from HSP validation."""

import math
from itertools import islice

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from .chemistry import normalize_smiles
from .descriptors import descriptor_provenance
from .intensive import COUNT_NAMES, INTENSIVE_VERSION, intensive_features
from .schema import PolymerInput

MAX_EXPLICIT_UNITS = 256
_STEREO_UID = "_hansenkit_stereo_uid"
_DOUBLE_STEREO = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
}
_DOUBLE_DIRECTIONS = {Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT}


def _fragment(smiles, ports):
    if smiles == "*[H]":
        builder = Chem.RWMol()
        builder.AddAtom(Chem.Atom(0))
        builder.AddAtom(Chem.Atom(1))
        builder.AddBond(0, 1, Chem.BondType.SINGLE)
        mol = builder.GetMol()
        Chem.SanitizeMol(mol)
    else:
        mol = Chem.MolFromSmiles(normalize_smiles(smiles))
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(Chem.GetMolFrags(mol)) != 1 or len(dummies) != ports:
        raise ValueError("Fragment must be connected with the required number of ports")
    if any(
        a.GetDegree() != 1
        or a.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE
        or a.GetNeighbors()[0].GetAtomicNum() == 0
        for a in dummies
    ):
        raise ValueError("Ports need a single bond to a real atom")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("Radical chain fragments require an explicit radical assembly backend")
    if ports == 2:
        labels = sorted(a.GetIsotope() for a in dummies)
        if labels not in ([0, 0], [1, 2]):
            raise ValueError("Use two unlabeled ports or distinct isotope ports [1*] and [2*]")
        dummies.sort(key=lambda a: (a.GetIsotope(), a.GetIdx()))
    for i, atom in enumerate(dummies):
        atom.SetProp("_hansenkit_side", ("left", "right")[i] if ports == 2 else "cap")
    return mol


def _port(mol, side):
    result = [
        a
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 0
        and a.HasProp("_hansenkit_side")
        and a.GetProp("_hansenkit_side") == side
    ]
    if len(result) != 1:
        raise ValueError("Ambiguous chain endpoint")
    return result[0]


def _stereo_uid(atom):
    return atom.GetIntProp(_STEREO_UID)


def _check_double_stereo_roundtrip(mol, constraints):
    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(_stereo_uid(atom))
    params = Chem.SmilesParserParams()
    params.removeHs = False
    roundtrip = Chem.MolFromSmiles(Chem.MolToSmiles(mapped, isomericSmiles=True), params)
    if roundtrip is None:
        raise ValueError("Assembled stereochemistry cannot be serialized")
    by_uid = {a.GetAtomMapNum(): a.GetIdx() for a in roundtrip.GetAtoms()}
    known = set()
    for ends, refs, stereo in constraints:
        bond = roundtrip.GetBondBetweenAtoms(*(by_uid[i] for i in ends))
        if bond is None or bond.GetStereo() not in _DOUBLE_STEREO:
            raise ValueError("Specified source double-bond stereo was lost")
        actual_ends = [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()]
        actual_refs = list(bond.GetStereoAtoms())
        if actual_ends == list(reversed(ends)):
            actual_refs.reverse()
        elif actual_ends != ends:
            raise ValueError("Source double-bond endpoint mapping changed")
        flip = False
        for end, ref, actual in zip(ends, refs, actual_refs, strict=True):
            if ref != roundtrip.GetAtomWithIdx(actual).GetAtomMapNum():
                neighbors = roundtrip.GetAtomWithIdx(by_uid[end]).GetNeighbors()
                if ref not in [a.GetAtomMapNum() for a in neighbors]:
                    raise ValueError("Source stereo reference is no longer adjacent")
                flip = not flip
        actual_same = bond.GetStereo() in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)
        expected_same = stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS)
        if actual_same ^ flip != expected_same:
            raise ValueError("Source double-bond geometry changed")
        known.add(frozenset(ends))
    for bond in roundtrip.GetBonds():
        ends = frozenset((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))
        if bond.GetStereo() in _DOUBLE_STEREO and ends not in known:
            raise ValueError("An unspecified double bond acquired stereo during assembly")


def _join(left, right, left_side="right", right_side="left"):
    left, right = Chem.Mol(left), Chem.Mol(right)
    for i, atom in enumerate(list(left.GetAtoms()) + list(right.GetAtoms())):
        atom.SetIntProp(_STEREO_UID, i + 1)
    constraints = []
    for mol in (left, right):
        for bond in mol.GetBonds():
            if bond.GetStereo() in _DOUBLE_STEREO:
                constraints.append(
                    (
                        [_stereo_uid(bond.GetBeginAtom()), _stereo_uid(bond.GetEndAtom())],
                        [_stereo_uid(mol.GetAtomWithIdx(i)) for i in bond.GetStereoAtoms()],
                        bond.GetStereo(),
                    )
                )
    left_port, right_port = _port(left, left_side), _port(right, right_side)
    replacements = {
        _stereo_uid(left_port): _stereo_uid(right_port.GetNeighbors()[0]),
        _stereo_uid(right_port): _stereo_uid(left_port.GetNeighbors()[0]),
    }
    left_port.SetAtomMapNum(1)
    right_port.SetAtomMapNum(1)
    # molzip preserves tetrahedral neighbor parity. Transfer double-bond geometry
    # independently so reused SMILES directions cannot conflict at a junction.
    directional = False
    for mol in (left, right):
        for bond in mol.GetBonds():
            if bond.GetBondDir() in _DOUBLE_DIRECTIONS:
                bond.SetBondDir(Chem.BondDir.NONE)
                directional = True
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
    output = Chem.molzip(left, right)
    Chem.SanitizeMol(output)
    if not constraints and not directional:
        if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in output.GetAtoms()):
            # A cap can also make a tetrahedral center non-stereogenic.
            Chem.AssignStereochemistry(output, cleanIt=True, force=True)
        return output
    output.ClearComputedProps()
    Chem.SanitizeMol(output)
    Chem.AssignStereochemistry(output, cleanIt=True, force=True)
    by_uid = {_stereo_uid(a): a.GetIdx() for a in output.GetAtoms()}
    potential = {
        info.centeredOn
        for info in Chem.FindPotentialStereo(output, cleanIt=False)
        if info.type == Chem.StereoType.Bond_Double
    }
    kept = []
    for ends, original_refs, stereo in constraints:
        refs = [replacements.get(i, i) for i in original_refs]
        bond = output.GetBondBetweenAtoms(*(by_uid[i] for i in ends))
        if bond is None:
            raise ValueError("A source double bond was removed during assembly")
        classic_sp2 = {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()} <= {
            6,
            7,
        }
        if classic_sp2 and bond.GetIdx() not in potential:
            # A methyl/H cap can make the substituents identical.
            continue
        # Other explicitly supplied flags (e.g. hypervalent P=N) retain source
        # encoding only; the potential-stereo classifier cannot qualify them.
        write_refs = refs
        if _stereo_uid(bond.GetBeginAtom()) != ends[0]:
            write_refs = list(reversed(refs))
        bond.SetStereoAtoms(*(by_uid[i] for i in write_refs))
        bond.SetStereo(stereo)
        kept.append((ends, refs, stereo))
    Chem.SetDoubleBondNeighborDirections(output)
    _check_double_stereo_roundtrip(output, kept)
    return output


def assemble_sequence(repeat_units, left_cap=None, right_cap=None):
    """Sequence is explicit; None means an explicitly declared hydrogen cap for this helper."""
    units = tuple(islice(iter(repeat_units), MAX_EXPLICIT_UNITS + 1))
    if not units or len(units) > MAX_EXPLICIT_UNITS:
        raise ValueError(f"Explicit assembly needs 1–{MAX_EXPLICIT_UNITS} units")
    chain = _fragment(units[0], 2)
    for smiles in units[1:]:
        chain = _join(chain, _fragment(smiles, 2))
    # Explicit hydrogen replacement preserves stereo and mass; RemoveHs handles final writing.
    chain = _join(chain, _fragment("*[H]" if left_cap is None else left_cap, 1), "left", "cap")
    chain = _join(chain, _fragment("*[H]" if right_cap is None else right_cap, 1), "right", "cap")
    if any(a.GetAtomicNum() == 0 for a in chain.GetAtoms()) or len(Chem.GetMolFrags(chain)) != 1:
        raise ValueError("Assembly left an unconnected fragment or port")
    expected = Chem.RemoveHs(chain)
    Chem.AssignStereochemistry(chain, cleanIt=True, force=True)
    smiles = normalize_smiles(Chem.MolToSmiles(chain, isomericSmiles=True))
    actual = Chem.MolFromSmiles(smiles)
    expected_count = sum(b.GetStereo() in _DOUBLE_STEREO for b in expected.GetBonds())
    actual_count = sum(b.GetStereo() in _DOUBLE_STEREO for b in actual.GetBonds())
    has_stereo = expected_count or any(
        a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in expected.GetAtoms()
    )
    if expected_count != actual_count or (
        has_stereo
        and (
            not actual.HasSubstructMatch(expected, useChirality=True)
            or not expected.HasSubstructMatch(actual, useChirality=True)
        )
    ):
        raise ValueError("Final canonical assembly changed source stereochemistry")
    return smiles


def assemble_homopolymer(repeat_unit, n, left_cap=None, right_cap=None):
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("Explicit repeat count must be an integer")
    if not 1 <= n <= MAX_EXPLICIT_UNITS:
        raise ValueError(f"Explicit repeat count must be within 1–{MAX_EXPLICIT_UNITS}")
    return assemble_sequence((repeat_unit,) * n, left_cap, right_cap)


def _caps(polymer):
    caps = []
    for end in polymer.end_groups:
        if end.count_per_chain not in {1, 2}:
            raise ValueError("Fractional end-group populations need a distribution backend")
        caps.extend([end.smiles] * int(end.count_per_chain))
    return tuple((caps + [None, None])[:2])


def characterize_homopolymer(polymer: PolymerInput):
    """Exact residue/end mass plus bounded asymptotic feature evaluation, not an HSP estimate."""
    if len(polymer.repeat_units) != 1 or polymer.architecture not in {"linear", "unknown"}:
        raise ValueError("This envelope requires one linear homopolymer repeat")
    unit = polymer.repeat_units[0].smiles
    caps = _caps(polymer)
    small = [assemble_homopolymer(unit, n, *caps) for n in (1, 2)]
    masses = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in small]
    residue_mass = masses[1] - masses[0]
    end_offset = masses[0] - residue_mass
    if residue_mass <= 0:
        raise ValueError("Nonpositive repeat residue mass")
    mean_n = (polymer.mn_g_mol - end_offset) / residue_mass
    if mean_n < 1:
        raise ValueError("Mn is smaller than one capped repeat")
    issues = []
    if any(cap is None for cap in caps):
        issues.append("unknown_end_groups_hydrogen_capped_representatives")
    if polymer.architecture == "unknown":
        issues.append("unknown_architecture_linear_representatives")
    if any(a.GetFormalCharge() for a in Chem.MolFromSmiles(small[0]).GetAtoms()):
        issues.append("formal_charge_preserved_not_property_supported")
    if polymer.eo_po is not None:
        bare = Chem.MolFromSmiles(unit)
        for atom in bare.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
        identity = normalize_smiles(Chem.MolToSmiles(bare))
        identities = {
            normalize_smiles("*OCC*"): "eo",
            normalize_smiles("*OCC(C)*"): "po",
        }
        if polymer.eo_po.eo_mean and polymer.eo_po.po_mean:
            issues.append("mixed_eo_po_is_not_a_homopolymer")
        elif identity not in identities:
            issues.append("eo_po_metadata_not_mapped_to_repeat_unit")
        else:
            kind = identities[identity]
            declared = getattr(polymer.eo_po, f"{kind}_mean")
            other = getattr(polymer.eo_po, "po_mean" if kind == "eo" else "eo_mean")
            if other > 0:
                issues.append("eo_po_metadata_disagrees_with_repeat_unit")
            elif abs(residue_mass * declared + end_offset - polymer.mn_g_mol) > max(
                0.01 * polymer.mn_g_mol, 0.1
            ):
                issues.append("eo_po_mean_and_mn_inconsistent")
    counts = (4, 16, 64)
    representatives = [assemble_homopolymer(unit, n, *caps) for n in counts]
    records = [intensive_features(s) for s in representatives]
    n_counts = len(COUNT_NAMES)
    heavy = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in representatives])
    raw = np.stack(
        [
            np.r_[r.values[:n_counts], r.values[2 * n_counts : 2 * n_counts + 8], r.values[-1]] * h
            for r, h in zip(records, heavy, strict=True)
        ]
    )
    slope, heavy_slope = (raw[2] - raw[1]) / 48, (heavy[2] - heavy[1]) / 48
    offset, heavy_offset = raw[1] - 16 * slope, heavy[1] - 16 * heavy_slope
    if heavy_slope <= 0 or not np.allclose(raw[0], 4 * slope + offset, atol=1e-7, rtol=1e-9):
        raise ValueError("Local descriptor additivity did not hold across 4/16/64 units")
    # Counts are integers; remove floating-point division/multiplication noise before large N.
    slope[:n_counts] = np.rint(slope[:n_counts])
    offset[:n_counts] = np.rint(offset[:n_counts])
    slope[-1], offset[-1] = round(slope[-1]), round(offset[-1])

    def compose(raw_values, atom_number):
        densities = raw_values / atom_number
        if (densities[:n_counts] < -1e-10).any() or densities[-1] < -1e-10:
            raise ValueError("Negative structural population")
        c = np.maximum(densities[:n_counts], 0)
        return np.r_[
            c,
            np.sqrt(c),
            densities[n_counts : n_counts + 8],
            1 / atom_number,
            1 / np.sqrt(atom_number),
            max(densities[-1], 0),
        ]

    bulk = compose(slope, heavy_slope)
    bulk[-3:-1] = 0  # inverse atom count and its square root vanish in the infinite-chain limit.
    if mean_n <= 64:
        low = math.floor(mean_n)
        high = math.ceil(mean_n)
        lower = intensive_features(assemble_homopolymer(unit, low, *caps)).values
        upper = intensive_features(assemble_homopolymer(unit, high, *caps)).values
        values = lower + (mean_n - low) * (upper - lower)
        basis = "two_adjacent_counts_moment_interpolation"
    else:
        values = compose(mean_n * slope + offset, mean_n * heavy_slope + heavy_offset)
        basis = "validated_local_additivity_4_16_64_mean_moment_envelope"
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite chain envelope")
    return {
        "series_id": polymer.series_id,
        "feature_version": INTENSIVE_VERSION,
        "descriptor_provenance": descriptor_provenance(),
        "residue_mass_g_mol": residue_mass,
        "end_mass_offset_g_mol": end_offset,
        "number_mean_repeat_count": mean_n,
        "mn_g_mol": polymer.mn_g_mol,
        "features": values.tolist(),
        "bulk_limit_features": bulk.tolist(),
        "representative_smiles": representatives,
        "explicit_repeat_counts": counts,
        "mean_feature_basis": basis,
        "issues": issues,
        "group_coverage_fraction": min(r.group_coverage_fraction for r in records),
        "max_explicit_atoms": max(Chem.MolFromSmiles(s).GetNumAtoms() for s in representatives),
        "hsp_predictions": None,
        "physical_accuracy_validated": False,
        "orientation": "isotope [1*] to [2*], otherwise canonical port order",
        "caveat": (
            "Mean Mn does not determine a full chain distribution, tacticity or material phase."
        ),
    }
