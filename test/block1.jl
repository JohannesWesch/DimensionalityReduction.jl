
using PlotlyJS


# Generate a range of values for x and y
x = -10:1:10  # Example range, can be adjusted
y = -10:1:10  # Example range, can be adjusted

# Generate the z_data matrix
z_data = [ -xi - yi for xi in x, yi in y ]

layout = Layout(
    autosize=false,
    width=500,
    height=500,
)
p = plot(surface(z=z_data, x=x, y=y, surfacecolor="lightblue"), layout)

savefig(p, "test/block1.svg")
p
