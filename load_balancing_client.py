from otc_gleba_service_client.client.grpc_client import GRPCClient
from otc_gleba_service_client.client.pricer.vanilla_pricer import VanillaPricer
from otc_gleba_service_client.client.pricer.snowball_pricer import SnowballPricer
import asyncio
import random


class DynamicLoadBalancingClient:
    def __init__(self, server_configs):
        self.servers = list(server_configs.keys())
        self.clients = {server: GRPCClient(server) for server in self.servers}
        self.server_stats = {
            server: {"active_tasks": 0, "total_tasks": 0} for server in self.servers
        }
        # Don't create Semaphores here
        self.semaphores = {}
        self.server_stats_queue = asyncio.Queue()
        self.should_process_queue = True

    async def initialize(self, max_concurrent_tasks):
        # Create a Semaphore for each server
        self.semaphores = {
            server: asyncio.Semaphore(max_concurrent_tasks) for server in self.servers
        }
        asyncio.create_task(self.process_server_stats_queue())

    async def process_server_stats_queue(self):
        while self.should_process_queue:
            task = await self.server_stats_queue.get()
            server, action = task
            if action == "increment":
                self.server_stats[server]["active_tasks"] += 1
                self.server_stats[server]["total_tasks"] += 1
            elif action == "decrement":
                self.server_stats[server]["active_tasks"] -= 1
            self.server_stats_queue.task_done()

    async def get_least_busy_client(self):
        # Find the minimum number of active tasks
        min_active_tasks = min(
            self.server_stats[server]["active_tasks"] for server in self.servers
        )
        # Select servers with the minimum number of active tasks
        least_busy_servers = [
            server
            for server in self.servers
            if self.server_stats[server]["active_tasks"] == min_active_tasks
        ]
        # Pick a random server among the least busy ones
        least_busy_server = random.choice(least_busy_servers)
        # Wait until the Semaphore allows starting a new task
        await self.semaphores[least_busy_server].acquire()
        self.server_stats[least_busy_server]["active_tasks"] += 1
        self.server_stats[least_busy_server]["total_tasks"] += 1

        await self.server_stats_queue.put((least_busy_server, "increment"))

        return least_busy_server

    async def mark_server_as_idle(self, server):
        self.server_stats[server]["active_tasks"] -= 1
        await self.server_stats_queue.put((server, "decrement"))
        # Release the Semaphore to allow starting a new task
        self.semaphores[server].release()

    async def get_vanilla_pricer(self, server):
        return VanillaPricer(self.clients[server])

    async def get_snowball_pricer(self, server):
        return SnowballPricer(self.clients[server])

    def get_server_stats(self):
        return self.server_stats

    async def close(self):
        self.should_process_queue = False
        await self.server_stats_queue.join()
