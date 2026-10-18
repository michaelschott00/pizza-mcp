FROM python:3.12

ARG OPENCODE_VERSION=2.0.31
ARG OPENCODE_INSTALL_URL=https://opencode.ai/v2/install
ARG AGENT_USER=agent
ARG AGENT_UID=1000
WORKDIR /workspaces/project

# Optional: Install system packages
# ENV DEBIAN_FRONTEND=noninteractive
# COPY packages.txt .
# RUN apt update -y \
#     && xargs apt -y install < packages.txt \
#     && rm packages.txt \
#     && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm requirements.txt

# Install opencode (or other harnesses)
RUN --mount=type=cache,target=/tmp/install-cache \
    curl -fsSL ${OPENCODE_INSTALL_URL} -o /tmp/install-cache/install.sh \
    && HOME=/home/${AGENT_USER} bash /tmp/install-cache/install.sh
ENV PATH="${PATH}:/home/${AGENT_USER}/.opencode/bin/"

# Install Node.js from the official binary distribution
RUN --mount=type=cache,target=/tmp/install-cache \
    curl -fsSL https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-x64.tar.xz \
        -o /tmp/install-cache/node.tar.xz \
    && tar -xJf /tmp/install-cache/node.tar.xz -C /usr/local --strip-components=1 \
    && node --version && npm --version

# Install the MCP inspector globally
RUN npm install -g @modelcontextprotocol/inspector

# Add unprivileged agent user
RUN useradd -u ${AGENT_UID} -m ${AGENT_USER} \
    && chown -R ${AGENT_UID}:${AGENT_UID} /home/${AGENT_USER} \
    && mkdir -p /home/agent/.config && chown -R ${AGENT_UID}:${AGENT_UID} /home/agent/.config \
    && mkdir -p /home/agent/.local && chown -R ${AGENT_UID}:${AGENT_UID} /home/agent/.local \
    && mkdir -p /tmp/opencode && chown -R ${AGENT_UID}:${AGENT_UID} /tmp/opencode

# The MCP Inspector config is bind-mounted from the repository (see compose.yml)
# so servers and flags can be changed without rebuilding the image. Create the
# mount point here: a missing target would otherwise be created as a directory,
# which a file mount cannot use.
RUN mkdir -p /home/${AGENT_USER}/.mcp-inspector \
    && touch /home/${AGENT_USER}/.mcp-inspector/mcp.json \
    && chown -R ${AGENT_UID}:${AGENT_UID} /home/${AGENT_USER}/.mcp-inspector
USER ${AGENT_USER}

ENTRYPOINT ["/bin/bash"]
