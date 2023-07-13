import torch
import torchvision.transforms as trans
from sklearn import datasets
from skimage.transform import downscale_local_mean


def create_input_bounds(img: torch.Tensor, eps: float) -> torch.Tensor:

    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img:
            The image.
        eps:
           The maximum accepted epsilon perturbation of each pixel.
    Returns:
        A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = torch.clip((img - eps), 0, 16)
    bounds[..., 1] = torch.clip((img + eps), 0, 16)

    return bounds.view(-1, 2)


# noinspection PyShadowingNames
def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):

    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes.
    """

    with open(spec_path, "w") as f:

        f.write(f"; Mnist property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))")


def create_instances_csv(num_props: int = 15, path: str = "mnistfc_instances.csv"):

    """
    Creates the instances_csv file.

    Args:
        num_props:
            The number of properties.
        path:
            The path of the csv file.
    """

    nets = ["mnist-net_256x2.onnx",
            "mnist-net_256x4.onnx",
            "mnist-net_256x6.onnx"]

    props = [f"prop_{i}_0.03.vnnlib" for i in range(num_props)]
    props += [f"prop_{i}_0.05.vnnlib" for i in range(num_props)]

    with open(path, "w") as f:

        for net in nets:
            timeout = 120 if net == "mnist-net_256x2.onnx" else 300
            for prop in props:

                if net == nets[-1] and prop == props[-1]:
                    f.write(f"{net},{prop},{timeout}")
                else:
                    f.write(f"{net},{prop},{timeout}\n")


if __name__ == '__main__':

    num_images = 2
    epsilons = [0.01]

    dig_data = datasets.load_digits()
    images = dig_data.images[1:25]
    labels = dig_data.target[1:25]

    scaled = []
    for img in images:
        scaled.append(downscale_local_mean(img, (2,2)))

    convert_tensor = trans.ToTensor()

    for eps in epsilons:
        for i in range(num_images):

            # input dim 64
            image, label = convert_tensor(images[i]), labels[i]

            #input dim 16
            #image, label = convert_tensor(scaled[i]), labels[i]


            input_bounds = create_input_bounds(image, eps)

            # input dim 16
            #spec_path = f"benchmarks/digits/dim16/prop_{i}_{eps:.2f}.vnnlib"

            # input dim 64
            spec_path = f"benchmarks/digits/dim64/prop_{i}_{eps:.2f}.vnnlib"

            save_vnnlib(input_bounds, label, spec_path)

    # create_instances_csv()