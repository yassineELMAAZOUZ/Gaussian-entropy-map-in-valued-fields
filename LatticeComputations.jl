using Nemo
import AbstractAlgebra
import IterTools



K, t = PuiseuxSeriesField(QQ, 10, "t")

"""
"normal_form" computes the normal lower triangular representation of a lattice. \n
	\t Input:  A a non singular matrix representing the full rank lattice (A 2d array with Nemo Puiseux series).\n
	\t Output: normal_A a lower triangular matrix representing the same lattice as A.\n

	Example:
				A = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}([t^0    t^0     t^2    ;
																	 			 t^0    t       t^0    ;
																	 			 t^0    t^2     t^3   ])

				normalA = normal_form(A)

				julia> indep_A
				3×3 Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq},2}:
				 1+O(t^10)  0+O(t^10)    0+O(t^10)
				 1+O(t^10)  1+O(t^10)    0+O(t^10)
				 1+O(t^10)  1+t+O(t^10)  1+O(t^10)
"""
function normal_form(B)

	n,d = size(B)

	A = copy(B)

	switch = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}(undef,n,1)


	for i = 1:n

		j = argmin(valuation.(A[i,i:d])) + i - 1

		switch = A[1:n,i]


		A[1:n,i] = A[1:n,j]
		A[1:n,j] = switch

		val = valuation(A[i,i])
		A[1:n,i] = t^(val) .* (A[1:n,i] .// A[i,i])

		for k = i+1:d
			alpha = - A[i,k]//A[i,i]
			A[1:n,k] = A[1:n,k] + alpha .* A[1:n,i]
		end

	end

	return A[1:n,1:n]

end


"""
"indep_lattice" computes the unique maximal independence lattice of a given lattice. \n
	\t Input:  A a non singular matrix representing the full rank lattice (A 2d array with Nemo Puiseux series).\n
	\t Output: indep_A an array containing the valuations of the diagonal elements of the maximal independence lattice.\n

	Example:
				A = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}([t^0    t^0     t^2    ;
																	 			 t^0    t       t^0    ;
																	 			 t^0    t^2     t^3   ])

				indep_A = indep_lattice(A)

				julia> indep_A
				3×1 Array{Rational{Int64},2}:
				0//1
				0//1
				0//1

"""
function indep_lattice(B)

	A = normal_form(B)

	d = size(A)[1]

	indep_A = Array{Rational{Int64}}(undef,d,1)
	indep_A[d] = valuation(A[d,d])

	for i = 1:d-1
		k = d-i

		container = copy(A[1:d,k])

		for j = k+1:d
			val = valuation(A[j,j]) - valuation(container[j])

			if val >= 0

				container = t^val .* container - (t^val * container[j]//A[j,j]) .* A[1:d,j]

			end
		end

		indep_A[k] = valuation(container[k])

	end

	return indep_A

end




"""
"compute_Polynomial" computes the multilinear polynomial corresponding to the tropicaliztion of the Gaussian measure. \n
	\t Input:  A a non singular matrix representing the full rank lattice (A 2d array with Nemo Puiseux series).\n
	\t Output: A dictionary of super-modular coefficients (a coefficient for every monomial).\n

	Example:
				 A = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}([t^0    0t     0t    ;
																	 			 t^0    t^2    0t    ;
																	 			 t^0    t      t^2   ])

				 julia> compute_Polynomial(A)
				 Dict{Array{Int64,N} where N,Int64} with 8 entries:
				   [1, 3]    => 1
				   [3]       => 0
				   [1, 2, 3] => 4
				   [1, 2]    => 2
				   [1]       => 0
				   [2, 3]    => 1
				   [2]       => 0
				   Int64[]   => 0

"""
function compute_Polynomial(A)

	d = size(A)[1]

	coefs = Dict{Array{Int64},Int64}([]=>0)

	for I in IterTools.subsets(1:d,1)
		B = normal_form(A[I,1:d])
		coefs[I] = valuation(B[1,1])
	end

	for I in IterTools.subsets(1:d,2)
		B = normal_form(A[I,1:d])
		coefs[I] = valuation(B[1,1]) + valuation(B[2,2])
	end


	for k = 2:d

		for I in IterTools.subsets(1:d,k)

			J = I[2:k]

			B = normal_form(A[I,1:d])
			L = indep_lattice(B)

			coefs[I] = coefs[J] + L[1]

		end

	end


	return coefs

end




A = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}([t^0    0t     0t 	  0t      0t    0t   0t;
												 				 t^0    t      0t 	  0t      0t    0t   0t;
												 				 t^0    t^2    t^3    0t      0t    0t   0t;
																 t^0    t^3    t^2    t^4     0t    0t   0t;
																 t^0    t^2    t^2    t       t^3   0t   0t;
																 t^0    t      t^2    t^3     t^4   t^5  0t;
																 t^0    t      t+t^2  t^2     t^3   t^2  t^4])


B = Array{AbstractAlgebra.Generic.PuiseuxSeriesFieldElem{fmpq}}([t^0    t^0     t^2    ;
												 				 t^0    t       t^0    ;
												 				 t^0    t^2    t^3   ])


@time P = compute_Polynomial(A)


d = size(A)[1]


for I in IterTools.subsets(1:d)

	println(I , "=======>",P[I])
end
