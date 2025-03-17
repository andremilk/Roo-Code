import { DatabricksProvider } from "../databricks"
import { Anthropic } from "@anthropic-ai/sdk"
import axios from "axios"

// Mock axios
jest.mock("axios")
const mockedAxios = axios as jest.Mocked<typeof axios>

describe("DatabricksProvider", () => {
	let provider: DatabricksProvider

	beforeEach(() => {
		provider = new DatabricksProvider({
			databricksBaseUrl: "https://test.databricks.com",
			databricksApiKey: "test-key",
			databricksModelId: "test-model",
		})
	})

	describe("constructor", () => {
		it("should initialize with provided config", () => {
			expect(provider["baseUrl"]).toBe("https://test.databricks.com")
			expect(provider["token"]).toBe("test-key")
			expect(provider["modelId"]).toBe("test-model")
		})

		it("should use default model ID if not provided", () => {
			const defaultProvider = new DatabricksProvider({
				databricksBaseUrl: "https://test.databricks.com",
				databricksApiKey: "test-key",
			})
			expect(defaultProvider["modelId"]).toBe("databricks-meta-llama-3-1-8b-instruct")
		})
	})

	describe("createMessage", () => {
		const mockMessages: Anthropic.Messages.MessageParam[] = [
			{
				role: "user",
				content: "Hello",
			},
			{
				role: "assistant",
				content: "Hi there!",
			},
		]

		const systemPrompt = "You are a helpful assistant"

		it("should handle text messages correctly", async () => {
			const mockResponse = {
				data: {
					choices: [
						{
							message: {
								content: "Test response",
							},
						},
					],
					usage: {
						prompt_tokens: 10,
						completion_tokens: 5,
					},
				},
			}

			mockedAxios.post.mockResolvedValueOnce(mockResponse)

			const stream = provider.createMessage(systemPrompt, mockMessages)
			const chunks = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			expect(chunks.length).toBe(2)
			expect(chunks[0]).toEqual({
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
			})
			expect(chunks[1]).toEqual({
				type: "text",
				text: "Test response",
			})

			expect(mockedAxios.post).toHaveBeenCalledWith(
				"https://test.databricks.com/serving-endpoints/test-model/invocations",
				expect.objectContaining({
					messages: [
						{
							role: "system",
							content: systemPrompt,
						},
						{
							role: "user",
							content: "Hello",
						},
						{
							role: "assistant",
							content: "Hi there!",
						},
					],
				}),
				expect.objectContaining({
					headers: {
						"Content-Type": "application/json",
						Authorization: "Bearer test-key",
					},
				}),
			)
		})

		it("should handle API errors", async () => {
			const mockError = new Error("Databricks API error")
			mockedAxios.post.mockRejectedValueOnce(mockError)

			const stream = provider.createMessage(systemPrompt, mockMessages)

			await expect(async () => {
				for await (const chunk of stream) {
					// Should throw before yielding any chunks
				}
			}).rejects.toThrow("Databricks API error")
		})

		it("should throw if base URL is missing", async () => {
			const invalidProvider = new DatabricksProvider({
				databricksApiKey: "test-key",
			})

			const stream = invalidProvider.createMessage(systemPrompt, mockMessages)

			await expect(async () => {
				for await (const chunk of stream) {
					// Should throw before yielding any chunks
				}
			}).rejects.toThrow("Databricks base URL is required")
		})

		it("should throw if API token is missing", async () => {
			const invalidProvider = new DatabricksProvider({
				databricksBaseUrl: "https://test.databricks.com",
			})

			const stream = invalidProvider.createMessage(systemPrompt, mockMessages)

			await expect(async () => {
				for await (const chunk of stream) {
					// Should throw before yielding any chunks
				}
			}).rejects.toThrow("Databricks API token is required")
		})
	})

	describe("getModel", () => {
		it("should return correct model info", () => {
			const modelInfo = provider.getModel()
			expect(modelInfo.id).toBe("test-model")
			expect(modelInfo.info).toBeDefined()
			expect(modelInfo.info.maxTokens).toBe(4096)
			expect(modelInfo.info.contextWindow).toBe(8192)
			expect(modelInfo.info.supportsImages).toBe(false)
			expect(modelInfo.info.supportsPromptCache).toBe(false)
		})

		it("should use custom max tokens if provided", () => {
			const customProvider = new DatabricksProvider({
				databricksBaseUrl: "https://test.databricks.com",
				databricksApiKey: "test-key",
				modelMaxTokens: 2048,
			})
			const modelInfo = customProvider.getModel()
			expect(modelInfo.maxTokens).toBe(2048)
		})

		it("should use custom temperature if provided", () => {
			const customProvider = new DatabricksProvider({
				databricksBaseUrl: "https://test.databricks.com",
				databricksApiKey: "test-key",
				modelTemperature: 0.8,
			})
			const modelInfo = customProvider.getModel()
			expect(modelInfo.temperature).toBe(0.8)
		})
	})
})