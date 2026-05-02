import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { z } from "zod";

const REPLICATE_API_URL = "https://api.replicate.com/v1/predictions";

const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
if (!REPLICATE_API_TOKEN) {
  throw new Error("Missing env var: REPLICATE_API_TOKEN");
}

const REPLICATE_MODEL_VERSION = process.env.REPLICATE_MODEL_VERSION;
if (!REPLICATE_MODEL_VERSION) {
  throw new Error("Missing env var: REPLICATE_MODEL_VERSION");
}

const NUM_OUTPUTS = parseInt(process.env.NUM_OUTPUTS ?? "1", 10);
const ASPECT_RATIO = process.env.ASPECT_RATIO ?? "1:1";
const OUTPUT_FORMAT = process.env.OUTPUT_FORMAT ?? "webp";
const DISABLE_SAFETY_CHECKER = process.env.DISABLE_SAFETY_CHECKER === "true";

const server = new McpServer({
  name: "facetime-mcp-server",
  version: "1.0.0",
});

server.registerTool(
  "facetime_mcp",
  {
    description:
      "Generate an image from a prompt and return the direct image URL.",
    inputSchema: z.object({
      prompt: z.string().min(1),
    }),
  },
  async ({ prompt }) => {
    const body = {
      version: REPLICATE_MODEL_VERSION,
      input: {
        model: "dev",
        prompt,
        go_fast: false,
        lora_scale: 1,
        megapixels: "1",
        num_outputs: NUM_OUTPUTS,
        aspect_ratio: ASPECT_RATIO,
        output_format: OUTPUT_FORMAT,
        guidance_scale: 3,
        output_quality: 80,
        prompt_strength: 0.8,
        disable_safety_checker: DISABLE_SAFETY_CHECKER,
        extra_lora_scale: 1,
        num_inference_steps: 28,
      },
    };

    const createRes = await fetch(REPLICATE_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${REPLICATE_API_TOKEN}`,
        "Content-Type": "application/json",
        Prefer: "wait",
      },
      body: JSON.stringify(body),
    });

    if (!createRes.ok) {
      const errText = await createRes.text();
      return {
        content: [
          {
            type: "text",
            text: `Replicate API error ${createRes.status}: ${errText}`,
          },
        ],
      };
    }

    const prediction = (await createRes.json()) as {
      id: string;
      status: string;
      output?: string[] | null;
      error?: string | null;
      urls?: { get?: string };
    };

    if (
      prediction.status === "succeeded" &&
      Array.isArray(prediction.output) &&
      prediction.output.length > 0
    ) {
      const imageUrl = prediction.output[0].trim();
      return {
        content: [{ type: "text", text: imageUrl }],
        structuredContent: { image_url: imageUrl },
      };
    }

    if (prediction.status === "failed") {
      return {
        content: [
          {
            type: "text",
            text: `Prediction failed: ${prediction.error ?? "Unknown error"}`,
          },
        ],
      };
    }

    const pollUrl = prediction.urls?.get;
    if (!pollUrl) {
      return {
        content: [
          {
            type: "text",
            text: `Error: Prediction returned status "${prediction.status}" with no poll URL. Prediction id: ${prediction.id}`,
          },
        ],
      };
    }

    for (let i = 0; i < 60; i++) {
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const pollRes = await fetch(pollUrl, {
        headers: {
          Authorization: `Bearer ${REPLICATE_API_TOKEN}`,
        },
      });

      if (!pollRes.ok) {
        const errText = await pollRes.text();
        return {
          content: [
            {
              type: "text",
              text: `Poll request failed ${pollRes.status}: ${errText}`,
            },
          ],
        };
      }

      const polled = (await pollRes.json()) as {
        status: string;
        output?: string[] | null;
        error?: string | null;
      };

      if (
        polled.status === "succeeded" &&
        Array.isArray(polled.output) &&
        polled.output.length > 0
      ) {
        const imageUrl = polled.output[0].trim();
        return {
          content: [{ type: "text", text: imageUrl }],
          structuredContent: { image_url: imageUrl },
        };
      }

      if (polled.status === "failed") {
        return {
          content: [
            {
              type: "text",
              text: `Prediction failed during polling: ${polled.error ?? "Unknown error"}`,
            },
          ],
        };
      }
    }

    return {
      content: [
        {
          type: "text",
          text: "Error: Timed out after 120 seconds waiting for Replicate prediction to complete.",
        },
      ],
    };
  },
);

const app = express();
app.use(express.json());

app.post("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
    enableJsonResponse: true,
  });

  res.on("close", () => transport.close());

  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

app.get("/health", (_req, res) => {
  res.json({ status: "ok", server: "facetime-mcp-server" });
});

const port = parseInt(process.env.PORT ?? "3000", 10);
app.listen(port, () => {
  console.log(`facetime-mcp-server running on port ${port}`);
});
