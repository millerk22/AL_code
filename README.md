# Active Learning Code

Included are the following Python code files and accompanying Jupyter Notebooks for testing the Active Learning methods.

* `util.py` -- first iteration of the methods used for active learning models, **ONLY BINARY (-1,+1) CASE**
* `util_multi.py` -- next iteration of methods, generalized to allow for both the binary case, as well as multi class
* `AL-EEM.ipynb` -- (Active Learning - Expected Error Minimization) Jupyter notebook with initial tests of the binary methods contained in `util.py`, along with tests done for finding faster ways of computing different functions in `util.py`.
* `AL-EEM-Multi.ipynb` -- the Multi-class extension of the main methods created and tested in `AL-EEM.ipynb`. Doesn't include the other tests done in `AL-EEM.ipynb` since those changes were already implemented into the code that was used to build `util-multi.py`.
* `SearchSpaceReduction.ipynb` -- playground for working with our "Search Space" reduction techniques involving the graph Laplacian. **Currently only compatible with `util.py` functions, i.e. binary case**
