import pstats

# Load the profile data
p = pstats.Stats('profile_output.pstats')

# Print the top 10 functions by total time
p.strip_dirs().sort_stats('tottime').print_stats(10)
