# CAP Theorem

    - you can only pick 2

	•	Consistency + Availability (CA):
        •	Works only if the system does not encounter network partitions.
        •	Typically found in systems with a single-node or strongly centralized architecture.

	•	Consistency + Partition Tolerance (CP):
        •	Sacrifices availability during partitions.
        •	Ensures data consistency even if some nodes are unreachable.
        •	Example: Traditional relational databases.

	•	Availability + Partition Tolerance (AP):
        •	Sacrifices consistency during partitions to ensure the system remains available.
        •	Example: Many NoSQL databases like Cassandra and DynamoDB.