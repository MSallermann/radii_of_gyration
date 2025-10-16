envname=rgenv

micromamba create -n ${envname} -c conda-forge python conda conda-pack
micromamba activate ${envname}

pip install -r requirements.txt

conda pack -n ${envname}

scp ./${envname}.tar.gz MareNostrumTransfer:~