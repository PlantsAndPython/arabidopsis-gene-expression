import pandas as pd
from tqdm import tqdm
from time import perf_counter


def filter_FPKM(infile="../../data/gene_FPKM_transposed.parquet",
                outfile="../../data/gene_FPKM_transposed_UMR75.parquet",
                metafile="../../data/metadata_UMR75.csv"):
    '''Filter RNAseq samples using SampleID from metadata file
        - metadata file contains SampleIDs filtered by unique mapped rate. We only want RNAseq profiles corresponding to those samples for later.
    
        Parameters
        ----------
        infile : str, default : "../data/gene_FPKM_transposed.parquet"
            path to the RNAseq file.

        outfile : str, default : "../data/gene_FPKM_transposed_UMR75.parquet"
            Filename for the filtered RNAseq data

        metafile : str, default : "../data/metadata_UMR75.csv"
            Filename for the filtered metadata file

        Returns
        -------

        None

    '''
    tic = perf_counter()
    mdf = pd.read_csv(metafile)
    print(f"metadata shape: {mdf.shape}")
    sampleIDs = mdf["SampleID"].tolist()
    rnadf = pd.read_parquet(infile)
    print(f"RNAseq data shape before filtering: {rnadf.shape}")
    rnadf.query('SampleID in @sampleIDs', inplace=True)
    print(f"RNAseq data shape after filtering: {rnadf.shape}")
    rnadf = pd.merge(mdf, rnadf, on='SampleID')
    print("Final DataFrame Shape: {rnadf.shape}")
    rnadf.to_parquet(outfile, compression="gzip", index=False)
    print("Wrote filtered RNAseq data to file")
    print(f"Time elapsed: {perf_counter() - tic}")

    return None


def filter_and_transpose_FPKM(infile="../../data/gene_FPKM_200501.parquet",
                outfile="../../data/gene_FPKM_transposed_UMR75.parquet",
                metafile="../../data/metadata_UMR75.csv"):
    '''Filter RNAseq samples using SampleID from metadata file
        - metadata file contains SampleIDs filtered by unique mapped rate. We only want RNAseq profiles corresponding to those samples for later.
    
        Parameters
        ----------
        infile : str, default : "../data/gene_FPKM_200501.parquet"
            path to the RNAseq file.

        outfile : str, default : "../data/gene_FPKM_transposed_UMR75.parquet"
            Filename for the filtered RNAseq data

        metafile : str, default : "../data/metadata_UMR75.csv"
            Filename for the filtered metadata file

        Returns
        -------

        None

    '''
    tic = perf_counter()
    ext = outfile.split(".")[-1]
    print(f"outfile extension: {ext}")

    mdf = pd.read_csv(metafile)
    print(f"metadata shape: {mdf.shape}")
    print(f"metadata columns: {mdf.columns}")
    sampleIDs = (mdf["SampleID"].tolist()).insert(0, "Sample")

    # df_list = []
    # df_array = pd.DataFrame()
    # for df in tqdm(pd.read_csv(infile, low_memory=False,
    #                            usecols=sampleIDs, chunksize = 10000)):
    #     print(' --- Complete')
    #     df_list.append(df)

    # df_array = pd.concat(df_list)

    df_array = pd.read_parquet(infile)
    if infile.endswith(".csv"):
        print(f"Saving {infile} to parquet format")
        df_array.to_parquet(infile.replace(".csv", ".parquet"),
                            index=False, compression="gzip")

    df_transposed = df_array.set_index("Sample").transpose()
    df_transposed = df_transposed.rename_axis("SampleID")\
        .rename_axis(None, axis="columns").reset_index()

    print(f"Shape before transpose: {df_array.shape}")
    print(f"Shape after transpose: {df_transposed.shape}")
    print(f"Column names: {df_transposed.columns}")
    print(df_transposed["SampleID"])
    print(df_transposed.head(2))

    # Merge filtered and transposed FPKM data with meta data
    df_transposed = pd.merge(mdf, df_transposed, on='SampleID')
    print(f"Final DataFrame Shape: {df_transposed.shape}")

    if ext == "csv":
        df_transposed.to_csv(outfile+".csv", index=False)
        print("Wrote transposed dataframe to csv")
    elif ext == "parquet":
        df_transposed.to_parquet(outfile, compression="gzip", index=False)
        print("Wrote transposed dataframe to parquet")
    else:
        print("extension not recognized. Options: '.csv' and '.parquet'")

    print(f"time elapsed: {perf_counter() - tic}")

    return None



if __name__ == "__main__":
    print("Filtering RNAseq samples...")
    filter_and_transpose_FPKM()
    print("Done...")