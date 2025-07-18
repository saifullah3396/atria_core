import torch

from atria_core.types.generic.label import Label, LabelList

label = Label(value=torch.tensor(0), name="Test Label")
label_list = LabelList(
    value=[torch.tensor(1), torch.tensor(2)], name=["Label One", "Label Two"]
)

print(label)
print(label_list)
