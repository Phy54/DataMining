import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Xem tổng quan bộ dữ liệu
def overview(df):
    st.subheader("🔍 Xem tổng quan dữ liệu")
    st.write(f"Bộ dữ liệu bao gồm {df.shape[0]} dòng và {df.shape[1]} cột.")
    st.dataframe(df)

# Kiểm tra dữ liệu Null
def nullData(df):
    st.subheader("📈 Thống kê dữ liệu")

    # Kiểm tra dữ liệu Null
    numNull = sum(df.isnull().sum())
    if (numNull != 0):
        st.write(f"Có {numNull} dòng dữ liệu rỗng")
        st.write(df[df.isnull().any(axis=1)])
    else:
        st.write("Không có giá trị Null.")

# Kiểm tra dữ liệu trùng lặp
def duplicatedData(df):

    numDup = df.duplicated().sum()
    if (numDup > 0):
        st.write(f"Có {numDup} dòng dữ liệu bị trùng lặp.")
        st.dataframe(df[df.duplicated()])
    else:
        st.write("Không có dòng dữ liệu nào bị trùng.")

# Kiểm tra dữ liệu duy nhất   
def uniqueData(df):
    st.write("Số lượng giá trị khác nhau ở mỗi cột:")
    st.dataframe(df.nunique())

# Vẽ biểu đồ thể hiện số lượng giao dịch ở các địa điểm
def plotLocationByTransactionVolume(df):
    col1, col2, col3 = st.columns([0.5, 4, 0.5])

    all_locations = df['Location'].value_counts().head(43)

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(y=all_locations.index, x=all_locations.values, palette='viridis')
    
    ax.tick_params(axis='both', labelsize=8)
    plt.title('All Locations by Transaction Volume', fontsize=20)
    plt.xlabel('Transaction Volume', fontsize=18)
    plt.ylabel('Location', fontsize=18)

    with col2:
        st.pyplot(fig)

# Vẽ biểu đồ thể hiện số lượng giao dịch theo Transaction Amount
def plotTransactionAmount(df):
    col1, col2, col3, col4 = st.columns([0.5, 2, 2, 0.5])
    
    # Vẽ biểu đồ cột thể hiện số lượng
    fig1, ax1 = plt.subplots(figsize = (10, 10))

    sns.histplot(data=df, x="TransactionAmount", color='pink', kde=True)

    plt.xlabel("Transaction Amount", fontsize=18)
    plt.ylabel("Count", fontsize=18)

    plt.title("Histplot of Transaction Amount", fontsize=20)

    with col2:
        st.pyplot(fig1)

    # Vẽ biểu đồ hộp thể hiện giá trị
    fig2, ax2 = plt.subplots(figsize = (10, 10))

    sns.boxplot(y=df['TransactionAmount'], color='pink')

    plt.title('Box Plot of Transaction Amount', fontsize=20)
    plt.ylabel('Transaction Amount', fontsize=18)

    with col3:
        st.pyplot(fig2)

# Vẽ biểu đồ thể hiện tỷ lệ của Transaction Type
def plotTransactionType(df):
    debitcard_counts = df[df['TransactionType']=='Debit']['TransactionType'].count()
    creditcard_counts = df[df['TransactionType']=='Credit']['TransactionType'].count()
    
    col1, col2, col3, col4 = st.columns([0.5, 2, 2, 0.5])
    
    index_values = [debitcard_counts, creditcard_counts]
    index_labels = ['Debit', 'Credit']

    # Vẽ biểu đồ cột thể hiện số lượng
    fig1, ax1 = plt.subplots(figsize = (10, 10))

    plt.bar(index_labels, index_values, color=['pink', 'lightblue'])
    plt.xlabel('Transaction Type', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    ax1.tick_params(axis='both', labelsize=10)

    # Vẽ biểu đồ tròn thể hiện tỷ lệ
    fig2, ax2 = plt.subplots(figsize = (10, 5))

    plt.pie(index_values, labels = index_labels, autopct='%2.2f%%', colors=['pink', 'lightblue'])

    with col2:
        st.pyplot(fig1)

    with col3:
        st.pyplot(fig2)

# Vẽ biểu đồ thể hiện tỷ lệ của Channel
def plotChannel(df):
    col1, col2, col3, col4 = st.columns([0.5, 2, 2, 0.5])

    all_channels = df['Channel'].value_counts().head(3)
    channel_counts = df['Channel'].value_counts()

    # Vẽ biểu đồ cột thể hiện số lượng
    fig1, ax1 = plt.subplots(figsize = (10, 10))

    sns.barplot(x=all_channels.index, y=all_channels.values, palette=['pink', 'lightblue', 'lightgreen'])
    plt.xlabel("Channel", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.title("Barplot of Channel", fontsize=20)
    ax1.tick_params(axis='both', labelsize=10)

    with col2:
        st.pyplot(fig1)

    # Vẽ biểu đồ tròn thể hiện tỷ lệ
    fig2, ax2 = plt.subplots(figsize = (10, 5))

    plt.pie(channel_counts, labels=channel_counts.index, autopct='%2.2f%%', colors=['pink', 'lightblue', 'lightgreen'])
    plt.title('Pie Chart of Channel', fontsize=20)
    plt.ylabel('Channel', fontsize=18)

    with col3:
        st.pyplot(fig2)

# Vẽ biểu đồ thể hiện phân bố của Customer Age
def plotCustomerAge(df):
    col1, col2, col3 = st.columns([0.5, 4, 0.5])

    fig , ax = plt.subplots(figsize=(10, 5))

    sns.histplot(data=df, x="CustomerAge", color='pink', kde=True)

    plt.title('Distribution of Customer Age', fontsize=20)
    plt.xlabel('Customer Age', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    ax.tick_params(axis='both', labelsize=10)
    
    with col2:
        st.pyplot(fig)

# Vẽ biểu đồ thể hiện tỷ lệ của Customer Occupation
def plotCustomerOccupation(df):
    col1, col2, col3, col4 = st.columns([0.5, 2, 2, 0.5])

    customer_occupation_counts = df['CustomerOccupation'].value_counts()

    fig1 , ax1 = plt.subplots(figsize=(10, 10))

    sns.barplot(x=customer_occupation_counts.index, y=customer_occupation_counts.values, palette=['pink', 'lightblue', 'lightgreen', 'yellow'])
    plt.title('Distribution of Customer Occupation', fontsize=20)
    plt.xlabel('Customer Ocupation', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    ax1.tick_params(axis='both', labelsize=10)
    
    with col2:
        st.pyplot(fig1)

    # Vẽ biểu đồ tròn thể hiện tỷ lệ
    fig2, ax2 = plt.subplots(figsize = (10, 10))
    
    plt.pie(customer_occupation_counts, labels=customer_occupation_counts.index, autopct='%2.2f%%', colors=['pink', 'lightblue', 'lightgreen', 'yellow'])
    plt.title('Pie Chart of Customer Occupation', fontsize=20)
    plt.ylabel('CustomerOccupation', fontsize=18)

    with col3:
        st.pyplot(fig2)

# Vẽ biểu đồ thể hiện phân bố của Account Balance
def plotBalance(df):
    col1, col2, col3 = st.columns([0.5, 4, 0.5])

    fig , ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(data=df, x="AccountBalance", color='navy', shade=True)
    plt.title('Density Plot of Account Balance of the Customer', fontsize=20)
    plt.xlabel('Balance', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    ax.tick_params(axis='both', labelsize=10)
    
    with col2:
        st.pyplot(fig)

# Xử lý dữ liệu
def preprocessData(df):
    df = df[['TransactionAmount','AccountBalance', 'CustomerAge']]
    df['oldBalance'] = df['TransactionAmount'] + df['AccountBalance']
    df['TransactionRate'] = df['TransactionAmount']/df['oldBalance']

    X = df[['TransactionAmount', 'TransactionRate', 'CustomerAge']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    return df, scaled_data

# Vẽ k-distance plot để chọn eps
def plotKDistance(df, scaled_data, k = 6):
    # Tính toán khoảng cách k-NN
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    # Dùng k gần nhất = min_samples
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(scaled_data)
    distances, indices = nbrs.kneighbors(scaled_data)

    # Lấy khoảng cách tới điểm thứ k
    distances = np.sort(distances[:, k-1])
    plt.figure(figsize=(10,5))
    plt.plot(distances)
    plt.title("K-distance plot để chọn eps")
    plt.xlabel("Dữ liệu được sắp xếp")
    plt.ylabel(f"Khoảng cách tới hàng xóm thứ {k}")
    plt.grid(True)

    col1, col2, col3 = st.columns([0.5, 4, 0.5])
    with col2:
        st.pyplot(plt.gcf())

# Tính toán DBSCAN
def computeDBSCAN(df, scaled_data, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN
    # Chạy DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(scaled_data)
    df['DBSCAN_Cluster'] = dbscan_labels

    # Tính toán số lượng cluster
    num_clusters = df['DBSCAN_Cluster'].nunique() - (1 if -1 in dbscan_labels else 0)
    st.write(f"Số lượng cluster: {num_clusters}")
    return dbscan_labels

# Mapping các nhãn DBSCAN
def mapDBSCANLabels(df, dbscan_labels):
    # Tạo mapping cho các nhãn
    count = pd.Series(dbscan_labels).value_counts().to_dict()
    mapping = {}
    for i in count:
        if i != -1:
            if i == max(count, key=count.get):
                mapping[i] = "Normal"
            else:
                mapping[i] = f"Suspicious Group {i}"
        else:
            mapping[i] = "Fraund"
    df['DBSCAN_Cluster'] = dbscan_labels

    # Áp dụng mapping cho nhãn
    df['DBSCAN_Cluster'] = df['DBSCAN_Cluster'].map(mapping)

    return df

# Vẽ biểu đồ phân tán DBSCAN trên 2 chiều
def visualizeDBSCAN2D(df, scaled_data):
    
    # Chọn các cột để vẽ biểu đồ
    df_scaled = pd.DataFrame(scaled_data, columns=['TransactionAmount', 'TransactionRate', 'CustomerAge'])
    df_plot = df_scaled.copy()
    df_plot['DBSCAN_Cluster'] = df['DBSCAN_Cluster']

    # Giảm chiều dữ liệu xuống 2 chiều để trực quan hóa
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)  # hoặc 2 để trực quan hóa
    X_pca = pca.fit_transform(scaled_data)
    st.write("Giảm chiều dữ liệu từ", scaled_data.shape[1], "xuống", X_pca.shape[1])

    # Thêm các cột gốc vào df_plot
    cols = df_plot.columns
    for col in df.columns:
        if col in cols:
            df_plot[f'org_{col}'] = df[col]

    # Vẽ biểu đồ phân tán
    fig, ax = plt.subplots(figsize=(10, 10))

    hover_cols = [col for col in df_plot.columns] # chỉ định các cột để hiển thị khi di chuột

    fig = px.scatter(df_plot, 
                     x=X_pca[:, 0],
                     y=X_pca[:, 1],
                     color=df_plot['DBSCAN_Cluster'],
                     hover_data=hover_cols, 
                     title="DBSCAN Clustering on Transactions",
                     height=600,
                    )
    
    col1, col2, col3 = st.columns([0.25, 5, 0.25])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

# 
def plotAmountBalance(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.scatterplot(data=df, x='AccountBalance', y='TransactionAmount', hue='TransactionType', alpha=0.6)
    plt.title('Transaction Amount vs. Account Balance')
    plt.xlabel('Account Balance')
    plt.ylabel('Transaction Amount')
    plt.legend(title = 'TransactionType')
    ax.tick_params(axis='both', labelsize=10)

    col1, col2, col3 = st.columns([0.25, 5, 0.25])
    with col2:
        st.pyplot(fig)

# Vẽ biểu đồ phân tán DBSCAN trên 3 chiều
def visualizeDBSCAN3D(df, scaled_data):

    # Vẽ biểu đồ phân tán 3D
    fig = px.scatter_3d(df, scaled_data[:,1], scaled_data[:,2], scaled_data[:,0],
                        color=df['DBSCAN_Cluster'],
                        title="DBSCAN Clustering on Transactions",
                        height=600)

    col1, col2, col3 = st.columns([0.25, 5, 0.25])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

# Vẽ heatmap tương quan
def plotCorrelation(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Heatmap of Correlation", fontsize=20)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig)

def computeSilhouette(df, scaled_data, dbscan_labels):

    from sklearn.metrics import silhouette_score

    # Tính toán chỉ số Silhouette
    if len(set(dbscan_labels)) > 1:
        silhouette_avg = silhouette_score(scaled_data, dbscan_labels)
        st.write(f"Chỉ số Silhouette: {silhouette_avg:.2f}")
    else:
        st.write("Không thể tính toán chỉ số Silhouette vì chỉ có một cụm.")

def reportOutliers(df, dbscan_labels):
    # Tạo DataFrame chứa các điểm ngoại lai
    outliers = df[dbscan_labels == -1]
    st.write(f"Số lượng điểm ngoại lai: {len(outliers)}")
    st.write("Các điểm ngoại lai:")
    st.dataframe(outliers)
    
    # Thống kê các điểm đáng ngờ
    suspicious_groups = df[(dbscan_labels != -1) & (dbscan_labels != 0)]
    if len(suspicious_groups) > 0:
        st.write("Các nhóm đáng ngờ:")
        st.write(f"Số lượng nhóm đáng ngờ: {len(suspicious_groups)}")
        st.dataframe(suspicious_groups)
    
    # Thống kê các điểm bình thường
    normal_groups = df[dbscan_labels == 0]
    if len(normal_groups) > 0:
        st.write("Các nhóm bình thường:")
        st.write(f"Số lượng nhóm bình thường: {len(normal_groups)}")
        st.dataframe(normal_groups)



st.set_page_config(page_title="Dataset Visualizer", layout="wide")

st.title("📊 Dataset Visualizer")

# Upload CSV
# uploaded_file = st.file_uploader("📁 Tải lên file CSV", type=["csv"])
uploaded_file = "./bank_transactions_data_2.csv"

st.sidebar.title("DBSCAN Clustering")
menu = st.sidebar.radio("Chọn trang", ["Explored Data Analysis", "DBSCAN Visualize"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if menu == "Explored Data Analysis":

        # Tạo Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Xem dữ liệu", "📈 Thống kê mô tả", "📊 Biểu đồ phân phối", "🔥 Heatmap tương quan"])

        with tab1:
            overview(df)

        with tab2:
            nullData(df)
            duplicatedData(df)
            uniqueData(df)

        with tab3:
            st.subheader("Biểu đồ thể hiện giá trị của TransactionAmount")
            plotTransactionAmount(df)

            st.subheader("Biểu đồ thể hiện tỷ lệ của Transaction Type")
            plotTransactionType(df)

            st.subheader("Biểu đồ thể hiện phân bố số lượng giao dịch ở các địa điểm")
            plotLocationByTransactionVolume(df)

            st.subheader("Biểu đồ thể hiện tỷ lệ của Channel")
            plotChannel(df)

            st.subheader("Biểu đồ thể hiện phân bố của Customer Age")
            plotCustomerAge(df)

            st.subheader("Biểu đồ thể hiện tỷ lệ của Customer Occupation")
            plotCustomerOccupation(df)

            st.subheader("Biểu đồ thể hiện phân bố của Account Balance")
            plotBalance(df)

            st.subheader("Biểu đồ thể hiện phân bố của Transaction Amount và Account Balance")
            plotAmountBalance(df)

        with tab4:
            st.subheader("🔥 Biểu đồ tương quan (heatmap)")
            plotCorrelation(df)

    if menu == "DBSCAN Visualize":

        df, scaled_data = preprocessData(df)
        
        min_samples = int(st.text_input("Nhập min_samples", value=6, key="min_samples"))
        # Tính toán K-distance
        st.subheader("🔍 Tính toán K-distance để chọn eps")
        plotKDistance(df, scaled_data, k=min_samples)

        # Tính toán DBSCAN
        eps = float(st.text_input("Nhập eps", value=0.54, key="eps"))
        
        if eps != 0.0:
            st.subheader("🔍 Tính toán DBSCAN")
            dbscan_labels = computeDBSCAN(df, scaled_data, eps, min_samples)
            df = mapDBSCANLabels(df, dbscan_labels)

            st.subheader("🔍 Biểu đồ phân tán DBSCAN 2D")
            visualizeDBSCAN2D(df, scaled_data)

            st.subheader("🔍 Biểu đồ phân tán DBSCAN 3D")
            visualizeDBSCAN3D(df, scaled_data)

            st.subheader("🔍 Tính toán chỉ số Silhouette")
            computeSilhouette(df, scaled_data, dbscan_labels)

            st.subheader("🔍 Báo cáo các điểm ngoại lai")
            reportOutliers(df, dbscan_labels)


        
else:
    st.info("Vui lòng tải lên một file CSV.")

