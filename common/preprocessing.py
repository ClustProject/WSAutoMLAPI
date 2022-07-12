class Preprocessing:
    # 공백, null값 처리
    def remove_missing_values(df):
        for col in df.columns:
            if df[col].values.dtype == 'O':
                df[col] = df[col].str.replace(' ', '')

                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(method='bfill')

                else:
                    pass

            else:
                df[col] = np.where(df[col] < 0, 0, df[col])
                df[col] = df[col].fillna(0)

        return df


    # In[ ]:
    def combine_df(df, conzon_id="콘존ID", vds_id="VDS_ID", traffic = '교통량', date = '집계일자', time = '집계시분'):
        # 지점교통량 컬럼 추가
        if (traffic not in df.columns) & (vds_id in df.columns):
            traffic_df = df.merge(df_vdsPointTrafficFile,
                                  how = 'left')

            # conzon_id 및 vds_id 둘다 있을경우는 *vds_id 기준으로 병합
            if (conzon_id in df.columns) & (vds_id in df.columns):
                merge_conzon_vds_id_df = traffic_df.merge(df_etcVdsFile,
                                       how = 'left') 

                return merge_conzon_vds_id_df

            # vds_id 기준 병합
            if vds_id in df.columns:
                merge_vds_id_df = traffic_df.merge(df_etcVdsFile,
                                       how = 'left') 

                return merge_vds_id_df

        # 구간교통량 컬럼 추가
        if (traffic not in df.columns) & (conzon_id in df.columns):
            traffic_df = df.merge(df_vdsNodeTrafficFile,
                                  how = 'left')



            # conzon_id 기준 병합
            if conzon_id in df.columns:
                merge_conzon_id_df = traffic_df.merge(df_etcConzonFile,
                                           how = 'left')

                return merge_conzon_id_df
        else:
            pass


    def label_encode(df):
        list_of_df = []
        list_of_col = []
        for col in df.columns:
            if (df[col].values.dtype == 'O') & ('ID' not in col):
                label = df[col]
                le = LabelEncoder()
                le.fit(label)
                label_encoded = le.transform(label)
                list_of_df.append(label_encoded)
                list_of_col.append('encoded_' + col)
            else:
                pass
        result = pd.DataFrame(list_of_df).T
        result.columns = list_of_col
        concat_df = pd.concat([df, result], axis=1)
        return concat_df


    def convert_to_datetime(df, date='집계일자', time='집계시분', encode='encoded'):
        format = '%Y%m%d'
        format_day = '%Y%m%d%H%M%S'
        try:
            if (time in df.columns) & (date in df.columns) & (encode not in df.columns):
                df[time] = df[time].apply(lambda x: str(x).rjust(4, '0'))
                df[time] = df[time].apply(lambda x: str(x).ljust(6, '0'))
                df['집계날'] = df[date].astype(str) + df[time]
                df['집계일'] = pd.to_datetime(df[date], format=format)
                # df['집계일'] = df[date].apply(lambda x: pd.to_datetime(str(x),
                # format=format))
                df['집계날'] = pd.to_datetime(df['집계날'], format=format_day)
                # df['집계날'] = df['집계날'].apply(lambda x: pd.to_datetime(str(x),
                # format=format_day))
            if time in df.columns:
                df[time] = df[time].apply(lambda x: str(x).rjust(4, '0'))
                df[time] = df[time].apply(lambda x: str(x).ljust(6, '0'))

            if date in df.columns:
                # df['집계일'] = df[date].apply(lambda x: pd.to_datetime(str(x),
                # format=format))
                df['집계일자'] = pd.to_datetime(df[date], format='%Y%m%d')

            else:
                pass

        except Exception as e:
            print(e)
            return df

        return df