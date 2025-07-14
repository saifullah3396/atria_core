# from atria_core.types.generic._raw.label import Label

# label = Label(name="example_label", value=1)
# print(label.batched([label, label, label]).batch_size)
# print(label.batched([label, label, label]).to_tensor().batch_size)
# print(label.to_tensor().to_device("cpu"))
# print(label.to_tensor().to_device())

from atria_core.types.generic._raw.label import Label

label = Label(name="example_label", value=1)
print(label.pa_schema())
# image = Image(content=PIL.Image.new("RGB", (100, 100), color="red"))
# # print(image.pa_schema())
# print(
#     image,
#     image.size,
#     image.shape,
#     image.channels,
#     image.content,
#     image.height,
#     image.width,
# )
# image = image.batched([image, image, image]).to_tensor()
# print(
#     image.shape,
#     image.dtype,
#     image.channels,
#     image.size,
#     image.source_height,
#     image.source_width,
# )
# # print(label.batched([label, label, label]).batch_size)
# # print(label.batched([label, label, label]).to_tensor().batch_size)
# # print(label.to_tensor().to_device("cpu"))
# # print(label.to_tensor().to_device())
