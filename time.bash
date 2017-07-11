#!/bin/bash

set -e

num_trials="$1"
shift
output_path="$1"
shift

echo "$0:"
echo "Running $num_trials trials"
echo "Writing output to $output_path"

echo 'wallclock_seconds	system_cpu_seconds	user_cpu_seconds	cpu_percentage	exit_status	major_faults	minor_faults	swaps	context_switches	output_path	command' > "$output_path"
for i in $(seq 1 $num_trials)
do
    echo
    echo '$' "$@"
    /usr/bin/time -a -o "$output_path" -f "%e	%S	%U	%P	%x	%F	%R	%W	%c	$output_path	%C" "$@"
done

echo
