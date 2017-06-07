
import logging
import os
import pickle


from pystan import StanModel

logger = logging.getLogger(__name__)


class CacheableStanModel(object):

    def __init__(self, path):

        self._path = path
        with open(path, "r") as fp:
            self._model_code = fp.read()


    @property
    def path(self):
        return self._path


    @property
    def model_code(self):
        return self._model_code


    @property
    def model(self):
        return getattr(self, "_model", None)


    @property
    def fit(self):
        return getattr(self, "_fit", None)


    def _load_model(self, recompile=False, overwrite=False, **kwargs):
        """
        Load the model.

        :param recompile:
            Re-compile the model if it has already been compiled.

        :param overwrite:
            Overwrite the compiled model path if it already exists.
        """

        if self.model is not None and not recompile:
            return self.model


        compiled_path = "{}.compiled".format(self.path)

        while os.path.exists(compiled_path) and not recompile:

            with open(compiled_path, "rb") as fp:
                model = pickle.load(fp)


            assert self._model_code is not None

            if self._model_code != model.model_code:
                logger.warn(
                    "Pre-compiled model differs to the code in {}; "\
                    "recompiling model".format(self.path))
                recompile = True
                continue

            else:
                logger.info(
                    "Using pre-compiled model from {}".format(compiled_path))
                break

        else:
            logger.info("Compiling model from {}".format(self.path))

            model = StanModel(model_code=self.model_code)

            # Save the compiled model.
            if not os.path.exists(compiled_path) or overwrite:
                with open(compiled_path, "wb") as fp:
                    pickle.dump(model, fp, -1)

        self._model = model
        return model


    def _validate_stan_inputs(self, **kwargs):
        """
        Check the format of the initial values for the model. If a dictionary
        is specified and multiple chains are given, then the initial value will
        be re-cast as a list of dictionaries (one per chain).
        """

        # Copy the dictionary of keywords.
        kwds = {}
        kwds.update(kwargs)

        # Allow for a keyword that will disable any verification checks.
        if not kwds.pop("validate", True):
            return kwds

        # Check chains and init values.
        if "init" in kwds.keys() and isinstance(kwds["init"], dict) \
        and kwds.get("chains", 1) > 1:

            init, chains = (kwds["init"], kwds.get("chains", 1))
            logger.info(
                "Re-specifying initial values to be list of dictionaries, "\
                "allowing one dictionary per chain ({}). "\
                "Specify validate=False to disable this behaviour"\
                .format(chains))
            
            kwds["init"] = [init] * chains

        if kwargs.get("data", None) is None:
            try:
                self._data 
            except AttributeError:
                self._data, self._metadata = self._prepare_data()

            kwds["data"] = self._data

        return kwds


    def optimize(self, data=None, recompile=False, overwrite=False, **kwargs):
        """
        Optimize the model given the data. Keyword arguments are passed directly
        to the `StanModel.optimizing` method.

        :param data:
            A dictionary containing the required key/value pairs for the STAN
            model.

        :param recompile: [optional]
            Re-compile the model if it has already been compiled.

        :param overwrite: [optional]
            Overwrite the compiled model path if it already exists.
        """

        self._load_model(recompile=recompile, overwrite=overwrite, **kwargs)

        kwds = self._validate_stan_inputs(data=data, **kwargs)
        return self.model.optimizing(**kwds)


    def sample(self, data=None, chains=4, iter=2000, warmup=None, 
        recompile=False, overwrite=False, **kwargs):
        """
        Draw samples from the model. Keyword arguments are passed directly to
        `StanModel.sampling`.

        :param data:
            A dictionary containing the required key/value pairs for the Stan
            model.

        :param chains: [optional]
            Positive integer specifying the number of chains.

        :param iter:
            Positive integer specifying how many iterations for each chain
            including warmup.

        :param warmup: [optional]
            Positive integer specifying the number of warmup (aka burn-in)
            iterations. As warm-up also specifies the number of iterations used
            for step-size adaption, warmup samples should not be used for
            inference. Defaults to iter // 2.

        :param recompile: [optional]
            Re-compile the model if it has already been compiled.

        :param overwrite: [optional]
            Overwrite the compiled model path if it already exists.
        """

        self._load_model(recompile=recompile, overwrite=overwrite, **kwargs)

        kwds = self._validate_stan_inputs(
            data=data, chains=chains, iter=iter, warmup=warmup, **kwargs)

        self._fit = self.model.sampling(**kwds)
        return self._fit
