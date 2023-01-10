### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ cff368ff-96dd-49de-993f-0791281655a9
begin
	using LinearAlgebra
	using DataFrames
end

# ╔═╡ eea32906-f8ea-4c20-821a-8e9aa54f2216
md"
# 1D Electrostatic Plasma Particle-In-Cell Simulation

Based on the algorithms described in [Plasma Simluations By Example by Brieda](https://www.routledge.com/Plasma-Simulations-by-Example/Brieda/p/book/9781138342323)
"

# ╔═╡ a64ea179-6f7f-4281-b79f-99f63249cb26
md"
## Simulation Code
"

# ╔═╡ af2c4b4d-2043-4f41-8cd6-5d378e7a7158
struct ParticleType
    name::String
	charge::Float64
	mass::Union{Nothing, Float64}
	count::Int64 # how many particles of this type
	positions::Union{Nothing, Vector{Float64}}
    velocities::Union{Nothing, Vector{Float64}}
	kinetic_energy::Union{Nothing, Vector{Float64}}
	potential_energy::Union{Nothing, Vector{Float64}}
end

# ╔═╡ 8bb5667e-2f03-4435-92d0-782e10a779ad
begin
	const Q_E = 1.602176565e-19 # C, electron charge
	const ϵ_0 = 8.85418782e-12  # C/V/m, vacuum perm.
	const M_E = 9.10938215e-31 # kg, electron mass
end

# ╔═╡ 0e21e1a2-d494-4b44-b486-de5e7b9e7802
"""

Updates the `charge_density_field` using the positions of the particles

# Arguments
- `charge_density_field`: 1D charge density field with the value defined for each node. This will be where the results of this function are stored.
- `particle_types::ParticleType`: Different particles with their corresponding positions. The `...` means you can pass in multiple types of particles.

# Examples
```julia-repl
julia> compute_charge_density(charge_density_field, electrons, deuterons)
```

"""
function compute_charge_density(charge_density_field, particle_types...)
	
end

# ╔═╡ ae09fb78-967d-4dce-9382-40418c7ec0fe
"""
Solve for potential using poission's equation: ``\\nabla^2 ϕ =  -\\frac{ρ}{ϵ_0}``. 

Poisson's Equation can be represented using finite differences in 1D: 

`` \\frac{d^2ϕ}{dx^2}= \\frac{ϕ_{i-1} + 2 ϕ_i + ϕ_{i+1}}{(Δ x)^2} = - \\frac{ρ_i}{ϵ_0} ``

"""
function compute_potential(charge_density_field, dx, boundary="dirichlet", ϕ_0=0, ϕ_n=0)
	NUM_NODES = length(charge_density_field)
	if boundary == "dirichlet"
		# Solve Aϕ = b for ϕ
		A = [1; 1/dx^2 * ones(NUM_NODES-2); 1] .* Tridiagonal([ones(NUM_NODES-2); 0], [1; -2 * ones(NUM_NODES-2); 1], [0; ones(NUM_NODES-2)])
		b = [ϕ_0; -charge_density_field[2:NUM_NODES-1]/ϵ_0; ϕ_n]

		ϕ = A \ b
		
		# compute_potential_thomas(potential_field, charge_density_field, dx)
		# compute_potential_gauss_seidel(potential_field, charge_density_field, dx)
	elseif boundary == "periodic"
		
	elseif boundary == "neumann"
		throw(ArgumentError("Neumann boundary condition not yet implemented."))
	else
		throw(ArgumentError("Unknown boundary type: $boundary."))
	end

	return ϕ
end

# ╔═╡ 376920c8-831c-49c6-9162-0c9b72ace3de
"""
The 1D finite difference for computing electric field (central diff):

``E = - \\frac{\\partial ϕ}{\\partial x} \\approx -\\frac{ϕ_{i+1} - ϕ_{i-1}}{2 Δ x}``

However, the above doesn't work for dirichlet boundaries, so we need the following expression at the boundaries:

2nd order forward diff: `` - \\frac{\\partial ϕ}{\\partial x} \\approx - \\frac{-3 ϕ_{i} + 4 ϕ_{i+1} - ϕ_{i+2}}{2 Δ x} ``

2nd order backwards diff: `` - \\frac{\\partial ϕ}{\\partial x} \\approx - \\frac{3 ϕ_{i} - 4 ϕ_{i-1} + ϕ_{i-2}}{2 Δ x} ``
"""
function compute_electric_field(potential_field, dx, boundary="dirichlet")
	NUM_NODES = length(potential_field)
	electric_field = Vector{Float64}(undef, NUM_NODES)
	
	if boundary == "dirichlet"
		
		# Internal Nodes
		for i in 2:NUM_NODES-1
			electric_field[i] = - (potential_field[i+1] - potential_field[i-1])/(2*dx)
		end

		# Boundary Nodes
		electric_field[1] = -(-3*potential_field[1] + 4*potential_field[2] -potential_field[3])/(2*dx)
		electric_field[NUM_NODES] = -(3*potential_field[NUM_NODES] - 4*potential_field[NUM_NODES-1] + potential_field[NUM_NODES-2])/(2*dx)
		
	elseif boundary == "periodic"
		
	elseif boundary == "neumann"
		throw(ArgumentError("Neumann boundary condition not yet implemented."))
	else
		throw(ArgumentError("Unknown boundary type: $boundary."))
	end

	return electric_field
end

# ╔═╡ 259fb5ed-f727-408c-9533-999b6baddd6e
"""
Use Linear interpolation to get the value of the field at `position`
"""
function interpolate(x0, dx, position::Float64, field::Vector{Float64})
	l_i = (position - x0)/dx
	i = trunc(Int, l_i)
	di = l_i - i

	if i+1 > length(field)
		return last(field)
	end

	if i < 1
		return first(field)
	end
	
	return field[i]*(1-di) + field[i+1]*(di)
end

# ╔═╡ 17084945-7a1f-47aa-b4fd-8fe37ab6b750
"""

Updates the velocities, positions, and energies of the particles using the electric field. Uses the leap frog method (kick-drift-kick).

# Arguments
- `x0`: Start of grid
- `dx`: The grid spacing
- `dt`: The time interval
- `electric_field`: 1D electric field with the value defined for each node
- `potential_field`: 1D electric field with the value defined for each node
- `ϕ_max`: Reference for potential calculations
- `particle_types::ParticleType`: The different types of particles for which we are tracking the position and velocity. The `...` means you can pass in multiple types of particles. Each particle type can have multiple particles.

"""
function update_particle_position_and_velocity(x0, dx, dt, electric_field, potential_field, ϕ_max, particle_types::ParticleType...)
	
	for particle_type in particle_types
		positions = particle_type.positions
		velocities = particle_type.velocities
		potential_energy = particle_type.potential_energy
		kinetic_energy = particle_type.kinetic_energy

		mass = particle_type.mass
		charge = particle_type.charge
		
		# Update the ith particle
		for i in 1:length(positions)
			old_position = positions[i]
			
			acceleration = (charge/mass)*interpolate(x0, dx, positions[i], electric_field)
			velocities[i] += dt * acceleration # kick
			positions[i] += dt * velocities[i] # drift

			ϕ = interpolate(x0, dx, (positions[i] + old_position)/2, potential_field)
			kinetic_energy[i] = 1/2*mass*velocities[i]*velocities[i]/Q_E # eV
			potential_energy[i] = charge*(ϕ-ϕ_max)/Q_E; # eV

		end
	end
end

# ╔═╡ 7f390b89-abd4-4d84-87b4-0a301f002b80
"""

Rewinds the velocities. This is necessary for the leapfrog method.

# Arguments
- `dx`: The grid spacing
- `dt`: The time interval
- `electric_field`: 1D electric field with the value defined for each node
- `particle_types::ParticleType`: The different types of particles for which we are tracking the position and velocity. The `...` means you can pass in multiple types of particles. Each particle type can have multiple particles.

"""
function velocity_rewind(x0, dx, dt, electric_field, particle_types::ParticleType...)
	for particle_type in particle_types
		velocities = particle_type.velocities
		positions = particle_type.positions
	
		# Update the ith particle
		for i in 1:length(positions)
			acceleration = (particle_type.charge/particle_type.mass)*interpolate(x0, dx, positions[i], electric_field)
			velocities[i] -= 1/2* dt * acceleration # kick
		end
	end
end

# ╔═╡ b4c9eec4-c17c-4150-be79-1922fc6bfc0e
md"## Simulation Examples"

# ╔═╡ 1a197459-08c0-4b30-a079-c1dcbe2af66e
md"### Example 1: Single electron in a region of uniform charge density between two grounded electrodes"

# ╔═╡ 8e13f1b9-5940-44d2-9bf1-e216791b0104
md"
Run the simulation with initial conditions set so that we have a constant potential of 0 on oppsosite ends of the 1D grid (grounded ends).

The positive ions make up the region of uniform charge density.

Assume that the positive ions do not move and only the electron moves. Assume we have constant charge density since positive ions don't move and electron contribution is negligible compared to contribution from positive ions.
"

# ╔═╡ e15475e9-99a1-41d2-9ca0-d15a9d667ed1
let
	# Initial Conditions
	NUM_NODES = 21
	NUM_TIMESTEPS = 4000
	x0 = 0.0 # starting grid coordinate on the x axis (i.e. origin)
	xm = 0.1 # ending grid coordinate on x-axis
	NUM_CELLS = NUM_NODES - 1 # number of mesh cells
	dx = (xm-x0)/(NUM_CELLS) # spacing between each cell/nodes
	dt = 1e-10 # time interval

	position_initial_electron = 4*dx
	velocity_initial_electron = 0
	electron_count = 1
	
	electron = ParticleType("electron", -Q_E, M_E, electron_count, fill(position_initial_electron, electron_count), fill(velocity_initial_electron, electron_count), Vector{Float64}(undef, electron_count), Vector{Float64}(undef, electron_count))
	
	positive_ions = ParticleType("positive ions", Q_E, nothing, 1e12*NUM_NODES, nothing, nothing, nothing, nothing)
	
	# Don't keep track of positions/velocities of positive ions since they're stationary, instead initialize the charge density field
	charge_density_field = fill(positive_ions.charge * positive_ions.count/NUM_NODES, NUM_NODES)::Vector{Float64}

	# Simulation initialization
	potential_field = compute_potential(charge_density_field, dx, "dirichlet", 0, 0)
	electric_field = compute_electric_field(potential_field, dx, "dirichlet")

	velocity_rewind(x0, dx, dt, electric_field, electron)

	ϕ_max = maximum(potential_field)

	df = DataFrame(TimeStep=Float64[], Position=Float64[], Velocity=Float64[], KineticEnergy=Float64[], PotentialEnergy=Float64[], TotalEnergy=Float64[])
	# Simulation Loop
	for timestep in 1:NUM_TIMESTEPS
	    # fields
		# No need to recompute fields since they're constant (we assumed constant charge density)
		# charge_density_field = compute_charge_density()
		# potential_field = compute_potential()
		# electric_field = compute_electric_field()
		
	    # particles
		update_particle_position_and_velocity(x0, dx, dt, electric_field, potential_field, ϕ_max, electron)

		
		# println(timestep*dt, ", ", electron.positions[1], ", ", electron.velocities[1], ", ", electron.potential_energy[1], ",", electron.kinetic_energy[1])

		push!(df, [timestep*dt, electron.positions[1], electron.velocities[1], electron.potential_energy[1], electron.kinetic_energy[1], electron.potential_energy[1]+ electron.kinetic_energy[1]])
	end

	df
end

# ╔═╡ 829be84d-2801-444b-a20f-a3aaefd326be
md"### Example 2: Two electron beams with opposite velocities (Two Stream-Instability)"

# ╔═╡ 4cc46a70-969b-4340-a0e7-fc61be15e9e7
# Initial Conditions


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
DataFrames = "~1.4.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.4"
manifest_format = "2.0"
project_hash = "4cb03024a2bf495e901bc4b72473bc6f99ac8fdc"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d4f69885afa5e6149d0cab3818491565cf41446d"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╟─eea32906-f8ea-4c20-821a-8e9aa54f2216
# ╟─a64ea179-6f7f-4281-b79f-99f63249cb26
# ╠═cff368ff-96dd-49de-993f-0791281655a9
# ╠═af2c4b4d-2043-4f41-8cd6-5d378e7a7158
# ╠═8bb5667e-2f03-4435-92d0-782e10a779ad
# ╠═0e21e1a2-d494-4b44-b486-de5e7b9e7802
# ╠═ae09fb78-967d-4dce-9382-40418c7ec0fe
# ╠═376920c8-831c-49c6-9162-0c9b72ace3de
# ╠═17084945-7a1f-47aa-b4fd-8fe37ab6b750
# ╠═259fb5ed-f727-408c-9533-999b6baddd6e
# ╠═7f390b89-abd4-4d84-87b4-0a301f002b80
# ╟─b4c9eec4-c17c-4150-be79-1922fc6bfc0e
# ╟─1a197459-08c0-4b30-a079-c1dcbe2af66e
# ╟─8e13f1b9-5940-44d2-9bf1-e216791b0104
# ╠═e15475e9-99a1-41d2-9ca0-d15a9d667ed1
# ╟─829be84d-2801-444b-a20f-a3aaefd326be
# ╠═4cc46a70-969b-4340-a0e7-fc61be15e9e7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
