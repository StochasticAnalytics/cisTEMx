---
title: "Architecture"
status: placeholder
description: "cisTEMx software architecture and design patterns"
content_type: explanation
audience: [developer, contributor]
level: advanced
topics: [api, contributing]
components: [gui, cli, database, algorithms]
---

# Architecture

Overview of cisTEMx software architecture, design patterns, and system organization.

=== "Basic"
    ## High-Level System Overview

    ```mermaid
    flowchart TB
        RD[Remote Desktop<br/>Access Layer]
        DS[Data Server<br/>Storage]
        CT[cisTEMx<br/>Core Application]
        USER((User))
        CP[Compute Resources<br/>Processing]

        RD --> CT
        DS --> CT
        CT --> USER
        CP <--> CT

        style CT fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
        style USER fill:#E8F4F8,stroke:#4A90E2,stroke-width:2px
        style RD fill:#F5F5F5,stroke:#999,stroke-width:2px
        style DS fill:#F5F5F5,stroke:#999,stroke-width:2px
        style CP fill:#F5F5F5,stroke:#999,stroke-width:2px
    ```

    **Key Components:**

    - **Remote Desktop**: User interface access point
    - **Data Server**: Cryo-EM image and metadata storage
    - **cisTEMx Core**: Central orchestration and processing logic
    - **Compute Resources**: Distributed processing for intensive calculations
    - **User**: Researcher interacting with the system

=== "Advanced"
    ## Component Architecture & Data Flow

    ```mermaid
    graph LR
        subgraph Client["Client Layer"]
            RD[Remote Desktop]
            GUI[GUI Frontend]
        end

        subgraph Core["cisTEMx Core"]
            APP[Application Logic]
            JM[Job Manager]
            DB[(Project Database)]
        end

        subgraph Storage["Data Layer"]
            DS[Data Server]
            MRC[MRC Files]
            STAR[STAR Metadata]
        end

        subgraph Compute["Compute Layer"]
            WN1[Worker Node 1]
            WN2[Worker Node 2]
            WNn[Worker Node N]
        end

        USER((User)) --> RD
        RD --> GUI
        GUI <--> APP
        APP <--> DB
        APP <--> DS
        DS --- MRC
        DS --- STAR
        JM --> WN1
        JM --> WN2
        JM --> WNn
        WN1 -.Results.-> JM
        WN2 -.Results.-> JM
        WNn -.Results.-> JM
        APP --> JM

        style Core fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
        style Storage fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
        style Compute fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
        style Client fill:#E8F5E9,stroke:#388E3C,stroke-width:2px
        style APP fill:#4A90E2,color:#fff
        style JM fill:#4A90E2,color:#fff
    ```

    **Architecture Layers:**

    - **Client Layer**: User access via Remote Desktop and GUI
    - **Core Layer**: Application orchestration, job management, project database
    - **Data Layer**: File storage with MRC images and STAR metadata
    - **Compute Layer**: Distributed worker nodes for parallel processing

=== "Developer"
    ## System Interaction & Communication Patterns

    ```mermaid
    sequenceDiagram
        participant U as User
        participant RD as Remote Desktop
        participant GUI as wxWidgets GUI
        participant Core as Core Application
        participant DB as SQLite Database
        participant JM as Job Manager
        participant DS as Data Server
        participant WN as Worker Nodes

        U->>RD: Connect via VNC/X11
        RD->>GUI: Launch Application
        GUI->>Core: Initialize
        Core->>DB: Load/Create Project
        DB-->>Core: Project Context

        Note over U,GUI: User initiates processing job
        U->>GUI: Submit Job
        GUI->>Core: Job Request
        Core->>DB: Store Job Parameters
        Core->>DS: Request Input Data
        DS-->>Core: MRC/STAR Files

        Note over Core,WN: Distributed Processing
        Core->>JM: Dispatch Job
        JM->>WN: Task Distribution
        WN->>WN: GPU/CPU Processing
        WN-->>JM: Partial Results
        JM-->>Core: Aggregated Results

        Core->>DB: Store Results
        Core->>DS: Write Output Files
        Core-->>GUI: Update UI
        GUI-->>U: Display Results

        Note over U,WN: Communication Protocols:<br/>Socket-based IPC (GUI-Core)<br/>SSH/File Transfer (Core-Workers)<br/>SQLite API (Core-DB)<br/>File I/O (Core-Storage)
    ```

    **Technical Details:**

    - **IPC Mechanism**: wxWidgets socket-based communication between GUI and processing threads
    - **Job Distribution**: SSH-based task execution on compute nodes
    - **Data Persistence**: SQLite for project metadata, file system for image data
    - **File Formats**: MRC for images, STAR for metadata, following RELION conventions
    - **Concurrency Model**: Multi-threaded GUI with separate process pool for compute tasks

---

!!! warning "Under Construction"
    This documentation is under development. Content will be migrated from developmental-docs repository.
