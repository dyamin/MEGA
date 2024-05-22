sampling_rate = 500  # hz

validity_threshold = 0
# list of valid movies will be chosen by this threshold.
#  This threshold represents the rate that a single movie classifier scored better than the naive score.
#  -1 stands for choosing the bad movies only (< 50%)
# if validity_threshold == 0:
#     valid_movies = "mov1, mov2, mov8, mov11, mov12, mov13, mov15, mov16, mov20, mov21," \
#                    " mov26, mov27, mov29, mov30, mov32, mov35, mov36, mov37, mov42, mov43, " \
#                    "mov45, mov47, mov51, mov52, mov53, mov54, mov55, mov56, mov57, mov58, " \
#                    "mov59, mov61, mov62, mov65, mov66, mov67, mov69, mov70, mov71, mov72, mov73, " \
#                    "mov74, mov75, mov76, mov77, mov78, mov79, mov80".split(", ")  # all movies
# elif validity_threshold == 50:
#     valid_movies = ['mov2', 'mov11', 'mov15', 'mov16', 'mov26', 'mov35', 'mov36', 'mov37',
#                     'mov53', 'mov54', 'mov55', 'mov56', 'mov57', 'mov58', 'mov65', 'mov66',
#                     'mov67', 'mov70', 'mov71', 'mov72', 'mov73', 'mov74', 'mov75', 'mov76',
#                     'mov78']  # > 50%
# elif validity_threshold == 60:
#     valid_movies = ['mov11', 'mov15', 'mov16', 'mov35', 'mov37', 'mov53', 'mov54', 'mov55',
#                     'mov57', 'mov58', 'mov65', 'mov66', 'mov70', 'mov71', 'mov72', 'mov73',
#                     'mov74', 'mov75', 'mov76']  # >= 60%
# elif validity_threshold == 70:
#     valid_movies = ['mov11', 'mov15', 'mov16', 'mov35', 'mov37', 'mov53', 'mov55', 'mov57',
#                     'mov65', 'mov66', 'mov70', 'mov71', 'mov72', 'mov73', 'mov75', 'mov76']  # > 70%
# elif validity_threshold == 80:
#     valid_movies = ['mov11', 'mov15', 'mov35', 'mov37', 'mov57', 'mov65', 'mov72', 'mov73',
#                     'mov75']  # >= 80%
# elif validity_threshold == 90:
#     valid_movies = ['mov11', 'mov35', 'mov37', 'mov65', 'mov72', 'mov73', 'mov75']  # > 90%
# elif validity_threshold == 100:
#     valid_movies = ['mov35', 'mov37']  # == 100%
# elif validity_threshold == -1:
#     valid_movies = ['mov1', 'mov2', 'mov8', 'mov12', 'mov13', 'mov20', 'mov21', 'mov26',
#                     'mov27', 'mov29', 'mov30', 'mov32', 'mov36', 'mov42', 'mov43', 'mov45',
#                     'mov47', 'mov51', 'mov52', 'mov54', 'mov56', 'mov58', 'mov59', 'mov61',
#                     'mov62', 'mov67', 'mov69', 'mov74', 'mov77', 'mov78', 'mov79', 'mov80']  # < 50%
