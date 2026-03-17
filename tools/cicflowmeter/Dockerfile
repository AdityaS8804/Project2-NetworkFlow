FROM --platform=linux/amd64 gradle:5.6-jdk8

RUN apt-get update && \
    apt-get install -y libpcap-dev maven git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone CICFlowMeter source
RUN git clone https://github.com/ahlashkari/CICFlowMeter.git . && \
    git checkout master

# Install jnetpcap to local Maven repo
RUN mvn install:install-file \
    -Dfile=jnetpcap/linux/jnetpcap-1.4.r1425/jnetpcap.jar \
    -DgroupId=org.jnetpcap \
    -DartifactId=jnetpcap \
    -Dversion=1.4.1 \
    -Dpackaging=jar

# Build distribution
RUN gradle distZip --no-daemon

# Unzip distribution in place
RUN mkdir -p /opt && cd /opt && \
    jar xf /app/build/distributions/CICFlowMeter-4.0.zip && \
    cp /app/jnetpcap/linux/jnetpcap-1.4.r1425/libjnetpcap.so /opt/CICFlowMeter-4.0/lib/native/ && \
    chmod +x /opt/CICFlowMeter-4.0/bin/cfm && \
    sed -i 's|../lib/native|/opt/CICFlowMeter-4.0/lib/native|g' /opt/CICFlowMeter-4.0/bin/cfm

# cfm <input_pcap> <output_dir>
ENTRYPOINT ["/opt/CICFlowMeter-4.0/bin/cfm"]
