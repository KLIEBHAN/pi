import { InMemoryModelsStore } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { ModelRuntime } from "../src/core/model-runtime.ts";
import type { ProviderChatModelConfig } from "../src/core/provider-composer.ts";

function model(id: string): ProviderChatModelConfig {
	return {
		id,
		name: id,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	};
}

function deferred() {
	let resolve = () => {};
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

// Regression coverage for https://github.com/earendil-works/pi/issues/11174.
describe("extension model refresh snapshots", () => {
	beforeEach(() => {
		vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Network disabled in snapshot tests"));
	});
	afterEach(() => vi.restoreAllMocks());

	it.each([false, true])("keeps known models throughout refresh (targeted: %s)", async (targeted) => {
		const runtime = await ModelRuntime.create({
			credentials: AuthStorage.inMemory(),
			modelsStore: new InMemoryModelsStore(),
			modelsPath: null,
		});
		const providerId = "extension-dynamic";
		let cached = [model("known")];
		let pauseOffline = false;
		let networkCalls = 0;
		const offlineStarted = deferred();
		const offlineRelease = deferred();
		const networkStarted = deferred();
		const networkRelease = deferred();
		runtime.registerProvider(providerId, {
			baseUrl: "https://example.test/v1",
			apiKey: "test-key",
			api: "openai-completions",
			refreshModels: async ({ allowNetwork }) => {
				if (!allowNetwork) {
					if (pauseOffline) {
						offlineStarted.resolve();
						await offlineRelease.promise;
					}
					return cached;
				}
				networkCalls++;
				networkStarted.resolve();
				await networkRelease.promise;
				cached = [model("known"), model("new")];
				return cached;
			},
		});
		await runtime.refresh({ allowNetwork: false });
		const ids = () =>
			runtime
				.getAvailableSnapshot()
				.filter((m) => m.provider === providerId)
				.map((m) => m.id);
		expect(ids()).toEqual(["known"]);
		const previousSnapshot = runtime.getAvailableSnapshot();
		pauseOffline = true;
		const refreshing = runtime.refresh({ allowNetwork: true, providers: targeted ? [providerId] : undefined });
		try {
			await offlineStarted.promise;
			expect(ids()).toEqual(["known"]);
			offlineRelease.resolve();
			await networkStarted.promise;
			expect(runtime.getModels(providerId).map((m) => m.id)).toEqual(["known"]);
			expect(ids()).toEqual(["known"]);
		} finally {
			offlineRelease.resolve();
			networkRelease.resolve();
			await refreshing;
		}
		expect(ids()).toEqual(["known", "new"]);
		expect(previousSnapshot.filter((m) => m.provider === providerId).map((m) => m.id)).toEqual(["known"]);
		expect(networkCalls).toBe(1);

		cached = [];
		await runtime.refresh({ allowNetwork: false });
		expect(ids()).toEqual([]);
		await runtime.refresh({ allowNetwork: false });
		expect(ids()).toEqual([]);
		expect(networkCalls).toBe(1);
	});

	it.each([false, true])("keeps known models when network refresh fails (aborted: %s)", async (aborted) => {
		const runtime = await ModelRuntime.create({
			credentials: AuthStorage.inMemory(),
			modelsStore: new InMemoryModelsStore(),
			modelsPath: null,
		});
		const controller = new AbortController();
		runtime.registerProvider("extension-dynamic", {
			baseUrl: "https://example.test/v1",
			apiKey: "test-key",
			api: "openai-completions",
			refreshModels: async ({ allowNetwork }) => {
				if (allowNetwork) {
					if (aborted) controller.abort();
					throw new Error("Refresh failed");
				}
				return [model("known")];
			},
		});
		await runtime.refresh({ allowNetwork: false });
		const result = await runtime.refresh({
			allowNetwork: true,
			providers: ["extension-dynamic"],
			signal: controller.signal,
		});
		expect(result.aborted).toBe(aborted);
		if (!aborted) expect(result.errors.get("extension-dynamic")?.message).toBe("Refresh failed");
		expect(
			runtime
				.getAvailableSnapshot()
				.filter((m) => m.provider === "extension-dynamic")
				.map((m) => m.id),
		).toEqual(["known"]);
	});

	it("does not carry published models into a different registration", async () => {
		const runtime = await ModelRuntime.create({
			credentials: AuthStorage.inMemory(),
			modelsStore: new InMemoryModelsStore(),
			modelsPath: null,
		});
		const config = {
			baseUrl: "https://example.test/v1",
			apiKey: "test-key",
			api: "openai-completions" as const,
			refreshModels: async () => [model("known")],
		};
		runtime.registerProvider("extension-dynamic", config);
		await runtime.refresh({ allowNetwork: false });
		expect(runtime.getModel("extension-dynamic", "known")).toBeDefined();

		runtime.registerProvider("extension-dynamic", { ...config, models: [model("replacement")] });
		expect(runtime.getModels("extension-dynamic").map((m) => m.id)).toEqual(["replacement"]);
		await runtime.refresh({ allowNetwork: false });

		runtime.unregisterProvider("extension-dynamic");
		await runtime.refresh({ allowNetwork: false });
		runtime.registerProvider("extension-dynamic", config);
		expect(runtime.getModels("extension-dynamic")).toEqual([]);
		await runtime.refresh({ allowNetwork: false });
		expect(runtime.getModel("extension-dynamic", "known")).toBeDefined();
	});
});
