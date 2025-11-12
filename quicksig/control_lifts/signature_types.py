from typing import NewType
from quicksig.hopf_algebras.elements import GroupElement, LieElement

# Backward-compatibility type names, now as newtypes over the new element classes.
# Runtime representation remains GroupElement/LieElement.
Signature = NewType("Signature", GroupElement)
LogSignature = NewType("LogSignature", LieElement)
BCKSignature = NewType("BCKSignature", GroupElement)
MKWSignature = NewType("MKWSignature", GroupElement)
