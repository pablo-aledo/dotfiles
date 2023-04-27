#!/bin/bash
#
#
# Bash Raytracer
# Herb Susmann
#

PIXEL_INCR="0.05"

# Object type IDs
SPHERE=0
PLANE=1

# Color IDs
BLACKID=0
REDID=1
GREENID=2

# Color ANSII escape codes
green="\033[92m"
red="\033[31m"
black="\033[30m"
endColor="\033[0m"


# call bc with arbitrary math expression
f() {
	bc <<< "scale=8; $1"
}

# vector dot product
vector_dot() {
	f "$1*$4+$2*$5+$3*$6"
}

intersect_sphere() {
	# Arguments:
	# $1 - sphere r
	# $2 - sphere x
	# $3 - sphere y
	# $4 - sphere z

	# $5 - ray origin x
	# $6 - ray origin y
	# $7 - ray origin z

	# $8 - ray direction x
	# $9 - ray direction y
	# $10 - ray direction z

	# a = d dot d
	a=$(vector_dot $8 $9 $10 $8 $9 $10)

	# dist = (o - c)
	dist=($(f "$5 - $2") $(f "$6 - $3") $(f "$7 - $4"))

	# dist_dot_d = (o - c) dot d
	dist_dot_d=$(vector_dot ${dist[0]} ${dist[1]} ${dist[2]} $8 $9 $10)

	# b = 2 * ((o - c) dot d)
	b=$(f "2.0 * $dist_dot_d")
	
	
	# dist_dot_dist = (o - c) dot (o - c)
	dist_dot_dist=$(vector_dot ${dist[0]} ${dist[1]} ${dist[2]} ${dist[0]} ${dist[1]} ${dist[2]})

	#c = (o - c) dot (o - c) - r^2
	c=$(f "$dist_dot_dist - ($1)^2")

	
	discriminant=$(f "$b * $b - 4 * $a * $c")

	if [ $(f "$discriminant < 0") -eq 1 ]
	then
		echo "0"
	else
		distance_sqrt=$(f "sqrt($discriminant)")
		t0=$(f "(-1 * $b - $distance_sqrt) / (2.0 * $a)")
		t1=$(f "(-1 * $b + $distance_sqrt) / (2.0 * $a)")

		if [[ $(f "$t0 > 0") -eq 1 && $(f "$t1 > 0") -eq 1 ]]
		then
			if [ $(f "$t0 < $t1") -eq 1 ]
			then
				echo $t1
			else
				echo $t0
			fi
		elif [[ $(f "$t0 > 0") -eq 1 && $(f "$t1 < 0") -eq 1 ]]
		then
			echo $t0
		elif [[ $(f "$t0 < 0") -eq 1 && $(f "$t1 > 0") -eq 1 ]]
		then
			echo $t1
		else
			echo "0"
		fi
	fi
}

intersect_plane() {
	# $1 - point x
	# $2 - point y
	# $3 - point z

	# $4 - normal x
	# $5 - normal y
	# $6 - normal z

	# $7 - ray origin x
	# $8 - ray origin y
	# $9 - ray origin z

	# $10 - ray direction x
	# $11 - ray direction y
	# $12 - ray direction z

	d_dot_n=$(vector_dot $10 $11 $12  $4 $5 $6)

	if [ $(f "$d_dot_n == 0") -eq 1 ]
	then
		echo "0"
	else
		# (p - o) * n
		p_minus_o_dot_n=$(vector_dot $4 $5 $6  $(f "$1 - $7") $(f "$2 - $8") $(f "$3 - $9"))
		t=$(f "$p_minus_o_dot_n / $d_dot_n")

		# If t is negative, we don't intersect. If positive, (or 0) we intersect.
		if [ $(f "$t < 0") -eq 1 ]
		then
			echo "0"
		else
			echo $t
		fi
	fi

}

# Print a horizontal line, the width of the output image (this is used for printing borders)
print_horizontal_line() {
	i=-1
	echo -n "$1"
	while [ $(f "$i <= 1") -eq 1 ]
	do
		echo -n "$1$1"

		i=$(f "$i + $PIXEL_INCR")
	done
	echo "$1"
}

objects=(
	$SPHERE 1.0 -0.4 0.0 0.0 $REDID
	$SPHERE 1.0 0.5 -2.0 1.0 $GREENID
	$PLANE  0.0 -1.0 -1.0 1.0 1.0 0.3 $GREENID
)

object_len=${#objects[@]}

focal_point=(0.0 0.0 -2.0)

# Print out a header
print_horizontal_line "▁"

z=-1
y=-1
while [ $( bc <<< "$y <= 1" ) -eq 1 ]
do
	echo -n "▏"
	x=-1
	while [ $( bc <<< "$x <= 1" ) -eq 1 ]
	do

		ray_direction=($(f "$x - ${focal_point[0]}") $(f "$y - ${focal_point[1]}") $(f "$z - ${focal_point[2]}"))

		i=0
		smallest_t=0
		closest_color=0
		while [ $i -lt $object_len ]
		do
			t=0
			if [ ${objects[$i]} -eq $SPHERE ]
			then
				t=$(intersect_sphere ${objects[$(($i + 1))]} ${objects[$(($i + 2))]} ${objects[$(($i + 3))]} ${objects[$(($i + 4))]} ${focal_point[0]} ${focal_point[1]} ${focal_point[2]} ${ray_direction[0]} ${ray_direction[1]} ${ray_direction[2]})
				if [ $(f "($t > 0 && $smallest_t == 0) || ($smallest_t > 0 && $t < $smallest_t && $t != 0)") -eq 1 ]
				then
					smallest_t=$t
					closest_color=${objects[$(($i + 5))]}
				fi

				i=$(($i + 6))
			elif [ ${objects[$i]} -eq $PLANE ] 
			then
				t=$(intersect_plane ${objects[$(($i + 1))]} ${objects[$(($i + 2))]} ${objects[$(($i + 3))]} ${objects[$(($i + 4))]} ${objects[$(($i + 5))]} ${objects[$(($i + 6))]} ${focal_point[0]} ${focal_point[1]} ${focal_point[2]} ${ray_direction[0]} ${ray_direction[1]} ${ray_direction[2]})

				if [ $(f "($t > 0 && $smallest_t == 0) || ($smallest_t > 0 && $t < $smallest_t && $t != 0)") -eq 1 ]
				then
					smallest_t=$t
					closest_color=${objects[$(($i + 7))]}
				fi

				i=$(($i + 8))
			fi

		done

		if [ $closest_color -eq $BLACKID ]
		then
			echo -n -e "${black}██${endColor}"
		elif [ $closest_color -eq $GREENID ]
		then
			echo -n -e "${green}██${endColor}"
		elif [ $closest_color -eq $REDID ]
		then
			echo -n -e "${red}██${endColor}"
		fi


		x=$(f "$x + $PIXEL_INCR")
	done

	echo "▕"

	y=$(f "$y + $PIXEL_INCR")
done

# Print footer
print_horizontal_line "▔"